import os.path as path
import pathlib

"""
Стоит запустить BEFORE_START для создания директории aig_test_benches, извлечения оригинальных bench из архива, 
или если надо очистить aig_benches от созданных в процессе файлов
"""

# Путь к директории, по которому будут сохраняться модели и файл мониторинга наград во время обучения
PATH_TO_MODELS_DIRECTORY = "Example/rl_model_example/"

# Путь к модели для (сохранения) и загрузки (из PATH_TO_MODELS_DIRECTORY)
PATH_TO_MODEL = pathlib.Path("Example/rl_model_example/model63.zip")

# Путь для сохранения модели DGI
BASE_DIR = path.dirname(path.abspath(__file__))
PATH_TO_SAVE_DGI = path.join(BASE_DIR, "Example", "dgi_model_example", "try16_8_state.pkl")

# Путь для создания и просмотра файла статистики модели в sb3_modelTest
PATH_TO_SAVE_STATS = "Example/rl_model_stats_example/"

# Путь для генерации csv файла статистик схем после 20 шагов (с названием)
PATH_FOR_GETTING_RANDOM_SCHEMES = pathlib.Path("Example/random_steps_stats/random_stats.csv")

# Путь к датасету DGI для Umap
PATH_TO_DGI_CSV = path.join(BASE_DIR, "common", "models_for_state", "umap_graphics", "dgi_embed_dataset.csv")

__all__ = [
        "PATH_TO_MODELS_DIRECTORY",
        "PATH_TO_MODEL",
        "PATH_TO_SAVE_DGI",
        "PATH_TO_SAVE_STATS",
        "PATH_FOR_GETTING_RANDOM_SCHEMES",
        "PATH_TO_DGI_CSV",
]
