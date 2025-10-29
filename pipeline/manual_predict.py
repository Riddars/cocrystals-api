import sys
import os
import datetime
import shutil
from pathlib import Path
from typing import List, Set

import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    str(import_path),
    str(Path(import_path) / 'TVAE' / 'generate'),
    os.getcwd()
])

# Хак для совместимости со старой версией allennlp, используемой в модели
import old_module
sys.modules['allennlp'] = sys.modules['old_module']
sys.modules['allennlp.modules'] = sys.modules['old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['old_module']

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from predict import main as run_ccgnet_prediction

# --- Конфигурация ---

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
        print(f"[INFO] Переменная BABEL_DATADIR установлена в: {babel_data_dir}")
else:
    print("[Предупреждение] Переменная CONDA_PREFIX не найдена. Убедитесь, что приложение запущено в активном conda-окружении.")


DRUG_SMILES: List[str] = [
    'CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O'
]

COFORMER_SMILES: List[str] = [
    'NC(=O)c1ccc(O)cc1',
    'Nc1ccc(C(=O)O)cc1',
    'O=C(O)c1ccc(O)cc1'
]

# --- Функции ---

def smiles_to_sdf_3d(smiles: str, output_path: Path) -> bool:
    """
    Конвертирует SMILES в SDF файл с 3D-координатами.

    Генерация 3D-структуры критична для корректной работы модели.

    Args:
        smiles: Строка SMILES молекулы.
        output_path: Путь для сохранения .sdf файла.

    Returns:
        True в случае успеха, False в случае ошибки.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [Ошибка] Не удалось создать молекулу из SMILES: {smiles}")
            return False

        mol = Chem.AddHs(mol)  # Водороды важны для корректной 3D геометрии
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        with Chem.SDWriter(str(output_path)) as writer:
            writer.write(mol)
        return True
    except Exception as e:
        print(f"  [Ошибка] Не удалось создать SDF для {smiles}. Ошибка: {e}")
        return False

def prepare_input_files(smiles_set: Set[str], prefix: str, target_dir: Path) -> List[str]:
    """
    Генерирует SDF файлы для набора уникальных SMILES.

    Args:
        smiles_set: Множество строк SMILES.
        prefix: Префикс для имен файлов (e.g., "drug", "coformer").
        target_dir: Директория для сохранения файлов.

    Returns:
        Список имен успешно созданных файлов.
    """
    created_files = []
    for i, smiles in enumerate(sorted(list(smiles_set))):
        filename = f"{prefix}_{i}.sdf"
        output_path = target_dir / filename
        if smiles_to_sdf_3d(smiles, output_path):
            created_files.append(filename)
    return created_files


def main():
    """Главная функция для запуска процесса предсказания."""
    # 1. Подготовка рабочей директории
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path.cwd() / f"manual_run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    print(f"Создана временная директория: {run_dir}\n")

    # 2. Генерация SDF файлов для лекарств и коформеров
    print("--- Шаг 1: Генерация 3D SDF файлов ---")
    unique_drugs: Set[str] = set(DRUG_SMILES)
    unique_coformers: Set[str] = set(COFORMER_SMILES)

    drug_files = prepare_input_files(unique_drugs, "drug", run_dir)
    coformer_files = prepare_input_files(unique_coformers, "coformer", run_dir)
    print("--- Генерация SDF файлов завершена. ---\n")

    if not drug_files or not coformer_files:
        print("[Ошибка] Не удалось создать SDF файлы для одной или обеих групп. Прерывание.")
        return

    # 3. Создание таблицы комбинаций (cc_table.csv)
    print("--- Шаг 2: Создание таблицы пар для анализа ---")
    table_path = run_dir / 'cc_table.csv'
    pair_count = 0
    with open(table_path, 'w') as f:
        for drug_file in drug_files:
            for coformer_file in coformer_files:
                # Формат: [файл лекарства]\t[файл коформера]\t[метка]
                # Метка (0/1) не важна для предсказания, используется как плейсхолдер.
                f.write(f"{drug_file}\t{coformer_file}\t1\n")
                pair_count += 1
    print(f"Создан файл '{table_path}' с {pair_count} парами для анализа.\n")

    # 4. Запуск предсказания CCGNet
    print("--- Шаг 3: Запуск предсказания CCGNet ---")
    print("Это может занять несколько минут...")
    output_xlsx_path = run_dir / 'ranked_cocrystal_predictions.xlsx'

    try:
        run_ccgnet_prediction(
            table=str(table_path),
            mol_dir=str(run_dir),
            fmt='sdf',
            model_type='cc',
            xlsx_name=str(output_xlsx_path)
        )
        print("\n--- УСПЕХ! ---")
        print("Предсказание завершено. Результаты сохранены в файле:")
        print(f"==> {output_xlsx_path} <==")

    except Exception as e:
        print("\n--- ОШИБКА ВО ВРЕМЯ ПРЕДСКАЗАНИЯ ---")
        print(f"Произошла ошибка: {e}")
        print("Пожалуйста, проверьте корректность настройки среды и установки зависимостей.")
        print(f"Промежуточные файлы находятся в папке: {run_dir}")

    # print(f"\nУдаление временной директории: {run_dir}")
    # shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()