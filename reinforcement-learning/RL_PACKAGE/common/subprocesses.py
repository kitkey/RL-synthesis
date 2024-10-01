import subprocess
import os
from os.path import exists
from pathlib import Path
import re


def run_seq_abc(name_with_d, commands, read_what="r"):
    """
    Базовая функция для работа с yosys-abc
    """
    sequence_of_commands = ["yosys-abc", f"-c {read_what} {name_with_d}; {commands}"]
    result = subprocess.check_output(sequence_of_commands, text=True)
    return result

def get_moment_aig(graph_scheme_name, i, worker_id):
    """
    Создание aig файла для оригинального bench файла схемы
    """
    path_mk = Path(f"common/aig_benches/{graph_scheme_name}/{worker_id}")
    if not exists(path_mk):
        os.mkdir(path_mk)

    finished_file_path_aig = f"common/aig_benches/{graph_scheme_name}/{worker_id}/{graph_scheme_name}_{i}.aig"
    file_path = f"common/aig_benches/{graph_scheme_name}/{graph_scheme_name}_orig.bench"
    run_seq_abc(file_path, f"strash; write_aiger {finished_file_path_aig}")


def extract_info(a):
    """
    Извлечение информации о схеме из строки
    """
    b = list(map(lambda x: x.split("/"), re.findall(r'=\d+/\d+', str(a[0]).replace(" ", ""))))[0]

    c = re.findall(r'=\d+', str(a[0]).replace(" ", ""))
    del c[0]

    d = re.findall(r'=\d+', str(a[1]).replace(" ", ""))
    del d[0]

    k = list(map(lambda x: x.replace("=", ""), (b + c + d)))
    k = list(map(int, k))

    inputs = k[0]
    outputs = k[1]
    ands = k[-2]
    edges = k[4]
    nodes = k[3]
    inverters = nodes - ands
    levels = k[-1]

    res = (inputs, outputs, ands, inverters, edges, levels)
    return res


def scheme_info(graph_scheme_name, i, worker_id):
    """
    Получение информации о схеме
    """
    aig_file = f"common/aig_benches/{graph_scheme_name}/{worker_id}/{graph_scheme_name}_{i}.aig"
    if i == 0:
        get_moment_aig(graph_scheme_name, 0, worker_id)
        file_path = f"common/aig_benches/{graph_scheme_name}/{graph_scheme_name}_orig.bench"
    else:
        file_path = f"common/aig_benches/{graph_scheme_name}/{worker_id}/{graph_scheme_name}_{i}.bench"
        sequence_of_commands1 = f"write_bench -l {file_path}"
        run_seq_abc(aig_file, sequence_of_commands1, "r")
    start_output = run_seq_abc(file_path, "ps", "r")

    final_output = run_seq_abc(aig_file, "ps", "read_aiger")
    a = (start_output, final_output)

    vector_info = extract_info(a)
    vector_state = vector_info

    return vector_state


def get_next_scheme(graph_scheme_name, i, action, worker_id):
    """
    Получение bench файла схемы после текущего шага
    """
    k = i + 1
    finished_file_path_aig = f"common/aig_benches/{graph_scheme_name}/{worker_id}/{graph_scheme_name}_{k}.aig"

    start_file_path = f"common/aig_benches/{graph_scheme_name}/{worker_id}/{graph_scheme_name}_{i}.aig"

    command_aig = f"strash; write_aiger {finished_file_path_aig}"
    command = action + "; " + command_aig

    run_seq_abc(start_file_path, command)


def clear_catalog():
    """
    Полное очищение директории aig_benches, остаются только папки с bench файлами оригинальных схем
    """
    path = Path(r"common/aig_benches")
    for f in path.rglob("**/*"):
        if f.is_file():
            if not str(f).endswith("orig.bench"):
                f.unlink()
    for f in path.rglob("**/*/*"):
        if f.is_dir():
            f.rmdir()
