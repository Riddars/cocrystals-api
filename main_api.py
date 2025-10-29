import sys
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Set

import_path = Path(__file__).parent.resolve()
sys.path.extend([
    str(import_path),
    str(import_path / 'TVAE' / 'generate'),
    os.getcwd()
])

# Хак для совместимости со старой версией allennlp
from pipeline import old_module
sys.modules['allennlp'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['pipeline.old_module']


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Локальные импорты
from pipeline.predict import main as run_ccgnet_prediction

# --- Конфигурация ---
RDLogger.DisableLog('rdApp.*')

# Путь к данным Open Babel. Убедитесь, что он корректен для вашей системы.
BABEL_DATA_DIR = r"D:\conda\envs\ENV_5\share\openbabel"
if not Path(BABEL_DATA_DIR).exists():
    print(f"[Предупреждение] Директория BABEL_DATADIR не найдена: {BABEL_DATA_DIR}")
else:
    os.environ['BABEL_DATADIR'] = BABEL_DATA_DIR

# --- Модели данных API ---
class PredictionRequest(BaseModel):
    drugs: List[str] = Field(
        ...,
        description="Список SMILES активных веществ (лекарств).",
        example=['CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O']
    )
    coformers: List[str] = Field(
        ...,
        description="Список SMILES коформеров-кандидатов.",
        example=['NC(=O)c1ccc(O)cc1', 'Nc1ccc(C(=O)O)cc1', 'O=C(O)c1ccc(O)cc1']
    )

# --- Вспомогательные функции ---

def _smiles_to_sdf_3d(smiles: str, output_path: Path) -> bool:
    """Конвертирует SMILES в SDF файл с 3D-координатами."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        with Chem.SDWriter(str(output_path)) as writer:
            writer.write(mol)
        return True
    except Exception:
        return False

def _prepare_input_files(smiles_set: Set[str], prefix: str, target_dir: Path) -> List[str]:
    """Генерирует SDF файлы для набора уникальных SMILES и возвращает список имен файлов."""
    created_files = []
    for i, smiles in enumerate(sorted(list(smiles_set))):
        filename = f"{prefix}_{i}.sdf"
        output_path = target_dir / filename
        if not _smiles_to_sdf_3d(smiles, output_path):
            raise ValueError(f"Некорректный или неконвертируемый SMILES ({prefix}): {smiles}")
        created_files.append(filename)
    return created_files

def _cleanup_temp_dir(path: Path):
    """Безопасно удаляет временную директорию."""
    if path.is_dir():
        shutil.rmtree(path)

# --- Основной пайплайн ---

def run_prediction_pipeline(drugs: List[str], coformers: List[str]) -> Path:
    """
    Выполняет весь процесс предсказания.
    Возвращает путь к сгенерированному .xlsx файлу.
    """
    run_dir = Path.cwd() / f"temp_run_{uuid.uuid4()}"
    run_dir.mkdir(exist_ok=True)

    try:
        # 1. Генерация SDF файлов
        drug_files = _prepare_input_files(set(drugs), "drug", run_dir)
        coformer_files = _prepare_input_files(set(coformers), "coformer", run_dir)

        # 2. Создание таблицы комбинаций
        table_path = run_dir / 'cc_table.csv'
        with table_path.open('w') as f:
            for drug_file in drug_files:
                for coformer_file in coformer_files:
                    f.write(f"{drug_file}\t{coformer_file}\t1\n")

        # 3. Запуск предсказания CCGNet
        output_xlsx_path = run_dir / 'ranked_cocrystal_predictions.xlsx'
        run_ccgnet_prediction(
            table=str(table_path),
            mol_dir=str(run_dir),
            fmt='sdf',
            model_type='cc',
            xlsx_name=str(output_xlsx_path)
        )

        if not output_xlsx_path.exists():
            raise RuntimeError("Модель предсказания не смогла создать выходной файл.")

        return output_xlsx_path

    except Exception:
        # При любой ошибке в пайплайне — немедленно удаляем временную папку
        _cleanup_temp_dir(run_dir)
        raise

# --- FastAPI приложение ---

app = FastAPI(
    title="GEMCODE Co-Crystal Predictor API",
    description="API для предсказания вероятности образования со-кристаллов.",
    version="1.0.0"
)

@app.post("/predict", response_class=FileResponse)
async def predict_cocrystals(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Принимает SMILES, запускает предсказание и возвращает Excel-файл.
    Временные файлы автоматически удаляются после отправки ответа.

    **Внимание:** Запрос может выполняться несколько минут. Убедитесь,
    что ваш клиент настроен на длительное ожидание ответа.
    """
    try:
        # Запуск тяжелого вычислительного процесса
        xlsx_path = run_prediction_pipeline(request.drugs, request.coformers)

        # Добавление задачи по очистке временной директории в фон.
        # Она выполнится ПОСЛЕ того, как ответ будет отправлен.
        temp_dir = xlsx_path.parent
        background_tasks.add_task(_cleanup_temp_dir, temp_dir)

        return FileResponse(
            path=str(xlsx_path),
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=xlsx_path.name
        )
    except ValueError as e:
        # Ошибка в данных, предоставленных пользователем (например, плохой SMILES)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Любая другая непредвиденная ошибка на сервере
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")