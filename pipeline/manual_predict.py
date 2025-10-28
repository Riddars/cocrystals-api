import sys
import os
import datetime
import shutil

# --- Начало блока инициализации среды (НЕ МЕНЯТЬ) ---
# Этот блок необходим для корректного импорта всех модулей проекта
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(import_path))
sys.path.append(str(import_path) + '/TVAE/generate')
sys.path.append(os.getcwd())

# Хак для совместимости со старой версией allennlp, используемой в модели
import old_module

sys.modules['allennlp'] = sys.modules['old_module']
sys.modules['allennlp.modules'] = sys.modules['old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['old_module']
# --- Конец блока инициализации среды ---

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # Отключаем лишние логи RDKit

# Импортируем главную функцию из predict.py
from predict import main as run_ccgnet_prediction

import os
os.environ['BABEL_DATADIR'] = r"D:\conda\envs\ENV_5\share\openbabel"

# ----------------- ВАШИ ДАННЫЕ ЗДЕСЬ -----------------

# Список SMILES активных веществ (лекарств)
# Дубликаты будут автоматически удалены
DRUG_SMILES_LIST = [
    'CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O'
]

# Список SMILES коформеров-кандидатов
COFORMER_SMILES_LIST = [
    'NC(=O)c1ccc(O)cc1',
    'Nc1ccc(C(=O)O)cc1',
    'O=C(O)c1ccc(O)cc1'
]


# ----------------------------------------------------


def smiles_to_sdf_3d(smiles, filename):
    """
    Конвертирует SMILES в SDF файл с 3D-координатами.
    Это КЛЮЧЕВОЙ шаг для исправления ошибки 'Bad input sample'.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [Ошибка] Не удалось прочитать SMILES: {smiles}")
            return False

        mol = Chem.AddHs(mol)  # Добавляем водороды, это важно для генерации 3D
        AllChem.EmbedMolecule(mol, randomSeed=42)  # Генерируем 3D-конформацию
        AllChem.UFFOptimizeMolecule(mol)  # Оптимизируем геометрию

        writer = Chem.SDWriter(filename)
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        print(f"  [Ошибка] Не удалось создать SDF для {smiles}. Ошибка: {e}")
        return False


def main():
    """
    Главная функция, которая выполняет весь процесс.
    """
    # 1. Создаем уникальную временную папку для чистоты эксперимента
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), f"manual_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Создана временная директория: {run_dir}\n")

    # Удаляем дубликаты из списков
    unique_drugs = sorted(list(set(DRUG_SMILES_LIST)))
    unique_coformers = sorted(list(set(COFORMER_SMILES_LIST)))

    # 2. Генерируем SDF файлы для всех уникальных молекул
    print("--- Шаг 1: Генерация 3D SDF файлов ---")
    drug_files = []
    for i, smiles in enumerate(unique_drugs):
        filename = os.path.join(run_dir, f"drug_{i}.sdf")
        if smiles_to_sdf_3d(smiles, filename):
            drug_files.append(os.path.basename(filename))

    coformer_files = []
    for i, smiles in enumerate(unique_coformers):
        filename = os.path.join(run_dir, f"coformer_{i}.sdf")
        if smiles_to_sdf_3d(smiles, filename):
            coformer_files.append(os.path.basename(filename))
    print("--- Генерация SDF файлов завершена. ---\n")

    # 3. Создаем cc_table.csv со всеми комбинациями
    print("--- Шаг 2: Создание таблицы пар для анализа ---")
    table_path = os.path.join(run_dir, 'cc_table.csv')
    pair_count = 0
    with open(table_path, 'w') as f:
        for drug_file in drug_files:
            for coformer_file in coformer_files:
                # Формат: [файл лекарства]\t[файл коформера]\t[метка]
                # Метка (0 или 1) здесь не важна для предсказания.
                f.write(f"{drug_file}\t{coformer_file}\t1\n")
                pair_count += 1
    print(f"Создан файл '{table_path}' с {pair_count} парами для анализа.\n")

    # 4. Запускаем предсказание с помощью CCGNet
    print("--- Шаг 3: Запуск предсказания CCGNet ---")
    print("Это может занять несколько минут...")
    output_xlsx_path = os.path.join(run_dir, 'ranked_cocrystal_predictions.xlsx')

    try:
        run_ccgnet_prediction(
            table=table_path,
            mol_dir=run_dir,
            fmt='sdf',
            model_type='cc',  # 'cc' для обычных со-кристаллов
            xlsx_name=output_xlsx_path
        )
        print("\n--- УСПЕХ! ---")
        print(f"Предсказание завершено. Результаты сохранены в файле:")
        print(f"==> {output_xlsx_path} <==")

    except Exception as e:
        print("\n--- ОШИБКА ВО ВРЕМЯ ПРЕДСКАЗАНИЯ ---")
        print(f"Произошла ошибка: {e}")
        print("Пожалуйста, проверьте, что ваша среда настроена корректно и все зависимости установлены.")
        print(f"Промежуточные файлы находятся в папке: {run_dir}")


if __name__ == "__main__":
    main()