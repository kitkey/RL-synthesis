from sb3_modelTest import model_test
from __init__ import PATH_TO_MODEL, PATH_TO_SAVE_STATS




if __name__ == "__main__":
    print("0 - if single scheme stat, 1 - if multiple schemes stats:")
    choose_test = int(input())
    if choose_test:
        model_test(path_to_model=PATH_TO_MODEL,
                   path_to_stats=PATH_TO_SAVE_STATS+"/stat.txt",
                   lstm=True,
                   test_num=1,
                   one_scheme=None)
    else:
        model_test(path_to_model=PATH_TO_MODEL,
                   path_to_stats=PATH_TO_SAVE_STATS + "/stat.txt",
                   lstm=True,
                   test_num=1,
                   one_scheme="testbench")