import sys
import os
import datetime
import shutil
import uuid  # Для создания уникальных имен папок

# --- Начало блока инициализации среды (КРИТИЧЕСКИ ВАЖНО!) ---
# Этот блок должен быть в самом верху, до импорта других модулей проекта.
# Он настраивает пути и хак для allennlp.
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(import_path))
# Добавляем пути к модулям, которые использует ваш проект
sys.path.append(os.path.join(import_path, 'TVAE', 'generate'))
sys.path.append(os.getcwd())

# Хак для совместимости со старой версией allennlp
from pipeline import old_module

sys.modules['allennlp'] = sys.modules['pipeline.old_module'] # Важно: нужно использовать полное имя модуля
sys.modules['allennlp.modules'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['pipeline.old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['pipeline.old_module']
# --- Конец блока инициализации среды ---

# Импорты для FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List

# Импорты из вашего проекта
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from pipeline.predict import main as run_ccgnet_prediction

import os
os.environ['BABEL_DATADIR'] = r"D:\conda\envs\ENV_5\share\openbabel"

# Отключаем лишние логи RDKit
RDLogger.DisableLog('rdApp.*')


# Важно: Указываем путь к данным Open Babel, если это необходимо
# Путь может отличаться в зависимости от системы
# os.environ['BABEL_DATADIR'] = r"D:\conda\envs\ENV_5\share\openbabel"


# --- Модели данных для FastAPI (определяем структуру JSON) ---

class PredictionRequest(BaseModel):
    drugs: List[str] = Field(...,
                             description="Список SMILES активных веществ (лекарств).",
                             example=[
                                 'CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O']
                             )
    coformers: List[str] = Field(...,
                                 description="Список SMILES коформеров-кандидатов.",
                                 example=['NC(=O)c1ccc(O)cc1', 'Nc1ccc(C(=O)O)cc1', 'O=C(O)c1ccc(O)cc1']
                                 )


# --- Логика предсказания (адаптировано из manual_predict.py) ---

def smiles_to_sdf_3d(smiles: str, filename: str) -> bool:
    """Конвертирует SMILES в SDF файл с 3D-координатами."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        with Chem.SDWriter(filename) as writer:
            writer.write(mol)
        return True
    except Exception:
        return False


def run_prediction_pipeline(drugs: List[str], coformers: List[str]) -> str:
    """
    Основная функция, выполняющая весь процесс предсказания.
    Возвращает путь к сгенерированному .xlsx файлу.
    """
    # 1. Создаем уникальную временную папку для каждого запроса
    run_dir = os.path.join(os.getcwd(), f"temp_run_{uuid.uuid4()}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        # 2. Генерируем SDF файлы для всех уникальных молекул
        unique_drugs = sorted(list(set(drugs)))
        unique_coformers = sorted(list(set(coformers)))

        drug_files = []
        for i, smiles in enumerate(unique_drugs):
            filename = os.path.join(run_dir, f"drug_{i}.sdf")
            if not smiles_to_sdf_3d(smiles, filename):
                raise ValueError(f"Некорректный SMILES лекарства: {smiles}")
            drug_files.append(os.path.basename(filename))

        coformer_files = []
        for i, smiles in enumerate(unique_coformers):
            filename = os.path.join(run_dir, f"coformer_{i}.sdf")
            if not smiles_to_sdf_3d(smiles, filename):
                raise ValueError(f"Некорректный SMILES коформера: {smiles}")
            coformer_files.append(os.path.basename(filename))

        # 3. Создаем cc_table.csv со всеми комбинациями
        table_path = os.path.join(run_dir, 'cc_table.csv')
        with open(table_path, 'w') as f:
            for drug_file in drug_files:
                for coformer_file in coformer_files:
                    f.write(f"{drug_file}\t{coformer_file}\t1\n")

        # 4. Запускаем предсказание с помощью CCGNet
        output_xlsx_path = os.path.join(run_dir, 'ranked_cocrystal_predictions.xlsx')

        run_ccgnet_prediction(
            table=table_path,
            mol_dir=run_dir,
            fmt='sdf',
            model_type='cc',
            xlsx_name=output_xlsx_path
        )

        if not os.path.exists(output_xlsx_path):
            raise RuntimeError("Модель предсказания не смогла создать выходной файл.")

        return output_xlsx_path

    except Exception as e:
        # Если что-то пошло не так, удаляем временную папку
        shutil.rmtree(run_dir)
        # И пробрасываем ошибку выше, чтобы FastAPI ее обработал
        raise e


# --- Создание FastAPI приложения и эндпоинта ---

app = FastAPI(
    title="GEMCODE Co-Crystal Predictor API",
    description="API для предсказания вероятности образования со-кристаллов.",
    version="1.0.0"
)


@app.post("/predict", response_class=FileResponse)
async def predict_cocrystals(request: PredictionRequest):
    """
    Принимает списки SMILES лекарств и коформеров, запускает пайплайн предсказания
    и возвращает результат в виде Excel-файла.

    **Внимание:** Этот запрос может выполняться несколько минут!
    Убедитесь, что ваш клиент настроен на длительное ожидание ответа.
    """
    try:
        # Запускаем тяжелый процесс предсказания
        xlsx_path = run_prediction_pipeline(request.drugs, request.coformers)

        # Возвращаем файл пользователю
        return FileResponse(
            path=xlsx_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=os.path.basename(xlsx_path)
        )
    except (ValueError, RuntimeError) as e:
        # Обрабатываем ожидаемые ошибки (например, неверный SMILES)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Обрабатываем непредвиденные ошибки на сервере
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")
    finally:
        # Важно! Этот блок не будет выполнен после `return`,
        # поэтому очистку нужно будет делать по-другому (см. ниже)
        pass


# Для правильной очистки файлов после отправки можно использовать Background Tasks
from fastapi import BackgroundTasks


@app.post("/predict_async", response_class=FileResponse)
async def predict_cocrystals_async(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Улучшенная версия эндпоинта, которая корректно удаляет временные файлы
    после отправки ответа клиенту.
    """

    def cleanup(path: str):
        shutil.rmtree(os.path.dirname(path))

    try:
        xlsx_path = run_prediction_pipeline(request.drugs, request.coformers)
        background_tasks.add_task(cleanup, path=xlsx_path)
        return FileResponse(
            path=xlsx_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=os.path.basename(xlsx_path)
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")