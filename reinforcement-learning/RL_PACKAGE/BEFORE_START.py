from zipfile import ZipFile
from os import mkdir
from os.path import exists, relpath
from common.subprocesses import clear_catalog

if __name__ == "__main__":
    path_test = relpath("common/aig_test_benches")
    if not exists(path_test):
        mkdir(path_test)
    path = relpath("common/aig_benches")
    if not exists(path):
        with ZipFile("common/aig_benches.zip") as f:
            f.extractall(path="common/")
    clear_catalog()