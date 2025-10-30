import sys
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict

import_path = Path(__file__).parent.resolve()
sys.path.extend([
    str(import_path),
    str(import_path / 'TVAE' / 'generate'),
    os.getcwd()
])
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
from pipeline.predict import main as run_ccgnet_prediction

RDLogger.DisableLog('rdApp.*')

# Получаем путь к текущему conda-окружению из переменной среды
conda_prefix = os.environ.get('CONDA_PREFIX')

if conda_prefix:
    # Строим правильный, кроссплатформенный путь к данным openbabel
    babel_data_dir = Path(conda_prefix) / 'share' / 'openbabel'

    if not babel_data_dir.exists():
        print(f"[Предупреждение] Директория данных Open Babel не найдена по ожидаемому пути: {babel_data_dir}")
    else:
        # Устанавливаем переменную среды, чтобы openbabel нашел свои файлы
        os.environ['BABEL_DATADIR'] = str(babel_data_dir)
        #print(f"[INFO] Переменная BABEL_DATADIR установлена в: {babel_data_dir}")
else:
    print("[Предупреждение] Переменная CONDA_PREFIX не найдена. Убедитесь, что приложение запущено в активном conda-окружении.")


# --- Модели данных API ---

class Job(BaseModel):
    """Описывает одно задание: одно лекарство и список его коформеров."""
    drug: str = Field(..., description="SMILES одного лекарственного вещества.")
    coformers: List[str] = Field(..., description="Список SMILES коформеров для этого лекарства.")

class PredictionRequest(BaseModel):
    """Основная модель запроса, которая содержит список заданий."""
    jobs: List[Job] = Field(
        ...,
        description="Список заданий для предсказания.",
        example=[
            {
                "drug": "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O",
                "coformers": ["NC(=O)c1ccc(O)cc1", "Nc1ccc(C(=O)O)cc1"]
            },
            {
                "drug": "CC(=O)Oc1ccccc1C(=O)O", # Аспирин
                "coformers": ["O=C(O)c1ccc(O)cc1"]
            }
        ]
    )

# --- Вспомогательные функции ---

# _smiles_to_sdf_3d остается без изменений
def _smiles_to_sdf_3d(smiles: str, output_path: Path) -> bool:
    """Конвертирует SMILES в SDF файл с 3D-координатами."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        with Chem.SDWriter(str(output_path)) as writer:
            writer.write(mol)
        return True
    except Exception:
        return False


# _cleanup_temp_dir остается без изменений
def _cleanup_temp_dir(path: Path):
    """Безопасно удаляет временную директорию."""
    if path.is_dir():
        shutil.rmtree(path)


# --- Основной пайплайн ---

def run_prediction_pipeline(jobs: List[Job]) -> Path:
    """
    Выполняет весь процесс предсказания на основе списка заданий.
    Возвращает путь к сгенерированному .xlsx файлу.
    """
    run_dir = Path.cwd() / f"temp_run_{uuid.uuid4()}"
    run_dir.mkdir(exist_ok=True)

    try:
        # 1. Собрать все уникальные SMILES из всех заданий
        all_smiles = set()
        for job in jobs:
            all_smiles.add(job.drug)
            all_smiles.update(job.coformers)

        # 2. Создать SDF файлы для всех уникальных молекул ОДИН РАЗ
        smiles_to_filename: Dict[str, str] = {}
        for i, smiles in enumerate(sorted(list(all_smiles))):
            filename = f"molecule_{i}.sdf"
            output_path = run_dir / filename
            if not _smiles_to_sdf_3d(smiles, output_path):
                raise ValueError(f"Некорректный или неконвертируемый SMILES: {smiles}")
            smiles_to_filename[smiles] = filename

        # 3. Создать таблицу комбинаций cc_table.csv на основе заданий
        table_path = run_dir / 'cc_table.csv'
        with table_path.open('w') as f:
            for job in jobs:
                drug_filename = smiles_to_filename[job.drug]
                for coformer_smiles in job.coformers:
                    coformer_filename = smiles_to_filename[coformer_smiles]
                    f.write(f"{drug_filename}\t{coformer_filename}\t1\n")

        # 4. Запуск предсказания CCGNet (эта часть не меняется)
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
        _cleanup_temp_dir(run_dir)
        raise

# --- FastAPI приложение ---

app = FastAPI(
    title="GEMCODE Co-Crystal Predictor API",
    description="API для предсказания вероятности образования со-кристаллов.",
    version="2.0.0"
)

@app.post("/predict", response_class=FileResponse)
async def predict_cocrystals(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Принимает список заданий (лекарство + его коформеры), запускает предсказание
    и возвращает Excel-файл.
    """
    try:
        xlsx_path = run_prediction_pipeline(request.jobs)

        temp_dir = xlsx_path.parent
        background_tasks.add_task(_cleanup_temp_dir, temp_dir)

        return FileResponse(
            path=str(xlsx_path),
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=xlsx_path.name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")