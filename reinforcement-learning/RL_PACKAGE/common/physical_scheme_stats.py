import os
import re
import subprocess

from common.subprocesses import run_seq_abc


def get_area_delay2(graph_name, passes_number, worker_id):
    c = run_seq_abc(f"common/aig_benches/{graph_name}/{worker_id}/{graph_name}_{passes_number}.aig",
                    "read common/nangate45.lib; strash; dch; map -B 0.9; topo; stime -c;  buffer -c; upsize -c; dnsize -c")  # map не те результаты выдает

    area = int(re.findall("Area=\d+", str(c).replace(" ", ""))[0][5::])
    delay = int(re.findall("Delay=\d+", str(c).replace(" ", ""))[0][6::])
    return {"area": area, "delay": delay}


def get_area_delay_test(graph_name, passes_number):
    c = run_seq_abc(f"common/aig_test_benches/{graph_name}/{graph_name}_step20_{passes_number}.bench",
                    "read common/nangate45.lib; strash; dch; map -B 0.9; topo; stime -c;  buffer -c; upsize -c; dnsize -c")

    area = int(re.findall("Area=\d+", str(c).replace(" ", ""))[0][5::])
    delay = int(re.findall("Delay=\d+", str(c).replace(" ", ""))[0][6::])
    return {"area": area, "delay": delay}