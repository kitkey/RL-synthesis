## common/subprocesses.py
В файле находятся скрипты автоматизации работы инструмента yosys-abc

## common/physical_scheme_stats
В файле находятся скрипты для получения площадей и задержек схем

## common/ADPstats.py
В файле находится скрипт для получения статистики в датасетах из площадей и задержек схем

## common/aig2graphml.py
В файле находится адаптированные из github openABC-Dataset скрипты для чтения bench файлов схем в формат networkx графов

## common/models_for_state/DGI_base_functions.py
Модель DeepGraphInfomax и необходиые для ее обучения функции. 

## common/models_for_state/DGI.py
Реализовано обучение модели DeepGraphInfomax

## common/models_for_state/DGI_umap.py
Генерация датасета из эмбеддингов графов, полученных с помощью DGI, и снижение их размерности для визуальной оценки работы модели

## common/custom_callback.py
В файле реализован callback для сохранения модели в случае длительного обучения без оценки работы модели

## __init__.py
В __init__.py хранятся пути к файлам, необходимым для работы скрипта 

## BEFORE_START.py
Скрипт, который следует запустить перед началом работы для распаковки папки с оригинальными bench файлами схем aig_benches и папки aig_test_benches, в которой будут создаваться 20 случайных последовательностей из 10 шагов для каждой схемы при запуске соответствующего скрипта из sb3_modelTest.py

## sb3_modelTest.py
В файле реализованы функция для сбора статистики model_test, для каждой схемы модель строит последовательность шагов оптимизации, выводится статистика для лучшего шага в этой последовательности и для конечного шага:
1) Выводится произведение площади на задержку схемы на лучшем шаге
2) Выводится произведение площади на задержку схемы на последнем шаге
3) Выводится последовательсность из всех шагов
4) Выводится статистика схем на лучшем и последнем шаге относительно среднего и лучшего результаты в датасете, с которым происходит сравнение, а также улучшение относительно оригинальной схемы
5) В конце файла выводится средняя статистика по всем схемам
Если аргументов в функции передается test_num = 0, результаты сравниваются с датасетом из 1500 последовательностей синтеза из openABC-Dataset, иначе - с кастомным датасетом, который необходимо предварительно сгенерировать функцией generate_test_schemes.

## eval_model.py
При запуске файла можно собрать статистику или одной схемы после применении последовательностей оптимизаций, сгенерированной моделью, или статистику схем после оптимизации по всему датасету

## sb3_modelTraining.py
Обучение модели алгоритмом RecurrentPPO из sb3-contrib
