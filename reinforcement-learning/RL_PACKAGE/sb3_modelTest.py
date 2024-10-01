import torch_sparse
import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import common.ADPstats as ADPstats

from common.osi_env import ICschemesEnv
from common.physical_scheme_stats import get_area_delay_test, get_area_delay2
from common.subprocesses import run_seq_abc
from __init__ import PATH_FOR_GETTING_RANDOM_SCHEMES, PATH_TO_SAVE_STATS

import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def model_test(path_to_model, path_to_stats, lstm=True, test_num=1, one_scheme=None):
    """ для test_num=1 следует сперва запустить функцию generate_test_schemes"""
    # seed_torch()
    env = ICschemesEnv(worker_index=1, reward_index=6)
    z = ""
    z += "custom test: " + str(test_num) + "\n\n"
    model = RecurrentPPO.load(path=path_to_model, print_system_info=False)

    scheme_list = ['spi', 'ss_pcm', 'usb_phy', 'des3_area', 'fpu', 'aes_xcrypt',
                   'tinyRocket', 'pci', 'simple_spi', 'aes', 'wb_dma', 'fir',
                   'tv80', 'aes_secworks', 'dynamic_node', 'sha256', 'ac97_ctrl', 'i2c',
                     'mem_ctrl', 'iir', 'sasc', 'wb_conmax']
    cus_list = ['ethernet', 'vga_lcd', 'jpeg', 'picosoc', 'dft', 'idft', 'bp_be']
    end_list = scheme_list + cus_list
    if one_scheme is not None:
        if one_scheme not in end_list:
            end_list = [one_scheme]
        else:
            end_list = [one_scheme]
            one_scheme = None

    sum_better_mean = 0
    sum_better_orig = 0
    sum_better_best = 0

    sum_better_mean_20 = 0
    sum_better_orig_20 = 0
    sum_better_best_20 = 0

    better_orig_percent_mean = 0
    better_orig_percent_mean_20 = 0
    better_mean_better_percent = 0
    better_mean_better_percent_20 = 0
    sum_succesful = 0
    sum_succesful_20 = 0
    lstm_states=None
    for graph_name in end_list:
        ad20_actions = []
        if lstm:
            lstm_states = None
        state, _ = env.reset(graph_n=graph_name)
        sum_reward = 0

        best_step = 0
        best_mul = 10 ** 16
        best_dict = {}
        ad20_dict = {}

        actions = []
        decode_action_numbers = {1: "b", 2: "rf", 3: "rw"}
        ad20_mul = 0
        for i in range(10):
            if lstm:
                action, lstm_states = model.predict(observation=state, state=lstm_states, deterministic=True)
                actions.append(decode_action_numbers[int(action) + 1])
                action = int(action)
            else:
                action, _ = model.predict(observation=state, deterministic=True)
                action = int(action)
                actions.append(decode_action_numbers[action + 1])
            state, reward, is_done, _, _ = env.step(action)

            ad_cur = get_area_delay2(graph_name, i+1, 1)
            adCur_mul = ad_cur["area"] * ad_cur["delay"]
            if adCur_mul < best_mul:
                best_mul = adCur_mul
                best_dict = ad_cur
                best_step = i + 1
            if i == 9:
                ad20_mul = adCur_mul
                ad20_dict = ad_cur
                ad20_actions = actions
            sum_reward += reward

        ad0 = get_area_delay2(graph_name, 0, 1)


        ad0_mul = ad0["area"] * ad0["delay"]
        if one_scheme is None:
            dicter = ADPstats.get_adp_stats(graph_name, test_number=test_num)
            mean_ad = dicter["mean area-delay"]
            best_ad = dicter["best area-delay"]
            better_orig = int(best_mul < ad0_mul)
            better_mean = int(best_mul < mean_ad)
            if test_num == 0:
                best_ad = best_ad[0]["ad"]
            elif test_num == 1:
                ...
            better_best = int(best_mul < best_ad)
            sum_succesful += int(int(best_mul < mean_ad) or int(best_mul < ad0_mul))

            better_orig_20 = int(ad20_mul < ad0_mul)
            better_mean_20 = int(ad20_mul < mean_ad)
            better_best_20 = int(ad20_mul < best_ad)
            sum_succesful_20 += int(int(ad20_mul < mean_ad) or int(ad20_mul < ad0_mul))


            sum_better_mean += better_mean
            sum_better_orig += better_orig
            sum_better_best += better_best

            sum_better_mean_20 += better_mean_20
            sum_better_orig_20 += better_orig_20
            sum_better_best_20 += better_best_20

            better_orig_percent = (1 - best_mul / ad0_mul)*100
            better_mean_percent = (1 - best_mul / mean_ad)*100
            better_best_percent = (1 - best_mul / best_ad)*100

            better_orig_percent_20 = (1 - ad20_mul / ad0_mul) * 100
            better_mean_percent_20 = (1 - ad20_mul / mean_ad) * 100
            better_best_percent_20 = (1 - ad20_mul / best_ad) * 100

            better_orig_percent_mean += better_orig_percent
            better_orig_percent_mean_20 += better_orig_percent_20
            better_mean_better_percent += better_mean_percent
            better_mean_better_percent_20 += better_mean_percent_20

            z += (
                    graph_name + "\norig area + delay: "
                    + str(ad0["area"]) + " " + str(ad0["delay"]) + " ad: " + str(ad0_mul)
                    + "\n" + f"{best_step} steps area + delay: "
                    + str(best_dict["area"]) + " " + str(best_dict["delay"]) + " ad: " + str(best_mul)
                    + "\n" + "10 steps area + delay: " + str(ad20_dict["area"]) + " " + str(ad20_dict["delay"]) + " ad: " + str(ad20_mul)
                    + f"\n{ad20_actions}\n\n"
            )


            z += (
                    "(unfixed) \nBetter than mean: {0} {1}%\n".format(better_mean, better_mean_percent)
                    + "Better than best : {0} {1}%\n".format(better_best, better_best_percent)
                    + "Better than orig: {0} {1}%\n\n".format(better_orig, better_orig_percent)
            )


            z += (
                    "(10 step) \nBetter than mean: {0} {1}%\n".format(better_mean_20, better_mean_percent_20)
                    + "Better than best : {0} {1}%\n".format(better_best_20, better_best_percent_20)
                    + "Better than orig: {0} {1}%\n\n".format(better_orig_20, better_orig_percent_20)
            )

            z += (
                "best ad: {0} mean ad: {1}\n\n".format(best_ad, mean_ad)
            )
            with open(path_to_stats, "a+") as f:
                f.write(z)
            z = ""
        else:
            z += (
                    graph_name + "\norig area + delay: "
                    + str(ad0["area"]) + " " + str(ad0["delay"]) + " ad: " + str(ad0_mul)
                    + "\n" + f"{best_step} steps area + delay: "
                    + str(best_dict["area"]) + " " + str(best_dict["delay"]) + " ad: " + str(best_mul)
                    + "\n" + "10 steps area + delay: " + str(ad20_dict["area"]) + " " + str(
                ad20_dict["delay"]) + " ad: " + str(ad20_mul)
                    + f"\n{ad20_actions}\n\n"
            )
            with open(path_to_stats, "a+") as f:
                f.write(z)
            z = ""


    if one_scheme is None:
        z += (
                "\n\n(unfixed step) \nBetter than orig: {0}\n".format(sum_better_orig)
                + "Better than orig mean: {0}%\n".format(better_orig_percent_mean/len(end_list))
                + "Better than mean (mean): {0}%\n".format(better_mean_better_percent / len(end_list))
                + "Better than mean: {0}\n".format(sum_better_mean)
                + "Better than best: {0}\n".format(sum_better_best)
                + "Good: {0}\n\n".format(sum_succesful)
        )

        z += (
                "(10 step) \nBetter than orig: {0}\n".format(sum_better_orig_20)
                + "Better than orig mean: {0}%\n".format(better_orig_percent_mean_20/len(end_list))
                + "Better than mean (mean): {0}%\n".format(better_mean_better_percent_20 / len(end_list))
                + "Better than mean: {0}\n".format(sum_better_mean_20)
                + "Better than best: {0}\n".format(sum_better_best_20)
                + "Good: {0}\n\n".format(sum_succesful_20)
        )

    with open(path_to_stats, "a+") as f:
        f.write(z)

def generate_test_schemes():
    """
    Следует запустить генерацию для датасета прежде чем собирать статистику модели model_test
    с параметром test_num = 1
    """
    scheme_list = ['spi', 'ss_pcm', 'usb_phy', 'des3_area', 'fpu', 'aes_xcrypt',
                   'tinyRocket', 'pci', 'simple_spi', 'aes', 'wb_dma', 'fir',
                   'tv80', 'aes_secworks', 'dynamic_node', 'sha256', 'ac97_ctrl', 'i2c',
                   'ethernet', 'mem_ctrl', 'iir', 'sasc', 'wb_conmax']

    cus_list = ['vga_lcd', 'jpeg', 'picosoc', 'dft', 'idft', 'bp_be']
    end_list = scheme_list + cus_list
    data = pd.DataFrame(columns=scheme_list)
    for name in end_list:
        commands = [';'.join(random.choices(['b', 'rf', 'rw'], k=10)) for _ in range(20)]
        os.mkdir(path=f"common/aig_test_benches/{name}")

        for i in range(10):
            command = 'strash; ' + commands[i] + f'; write_bench -l common/aig_test_benches/{name}/{name}_step10_{i+1}.bench'
            run_seq_abc(f"common/aig_benches/{name}/{name}_orig.bench", command)
            ad_dict = get_area_delay_test(name, i+1)
            data.at[i, name] = ad_dict["area"] * ad_dict["delay"]
        print(data)
    data.to_csv(path_or_buf=PATH_FOR_GETTING_RANDOM_SCHEMES)
