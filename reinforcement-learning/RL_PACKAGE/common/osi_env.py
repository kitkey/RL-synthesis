import random

import gymnasium as gym
from gymnasium import spaces

from __init__ import PATH_TO_SAVE_DGI
from common.subprocesses import get_moment_aig, get_next_scheme, scheme_info
from common.models_for_state.DGI_base_functions import DeepGraphInfomax, Encoder, corruption, networkx_load

import torch
import torch.nn as nn
import numpy as np

from common.physical_scheme_stats import get_area_delay2
from common.aig2graphml import *


def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





class ICschemesEnv(gym.Env):
    """
    Среда
    worker_index - Номер рабочего для параллельного обучения, чтобы работа с файлами происходила в отдельных папках,
    для одиночного обучения по умолчанию worker_index = 1
    reward_index - номер формула награды, даваемой рабочему:
        1 - награда дается после завершения эпизода из 20 шагов относительно начальной схемы
        2 - награда дается каждые passes_reward шагов относительно схемы passes_reward шагов назад и оригинальной схемы
        3 - награда дается после 20 шагов: +-1 в зависимости от их успешности
        4 - награда +-0.04 дается относительно 1 схемы каждый шаг, +-0.96 на последнем шаге
        5 - награда за каждый шаг считается относительно предыдущего и умножаетсян а 0.04, после последнего шага - на 0.96
        6 - тестовая награда (она не высчитывается)
        8 - награда считается относительно предыдущего состояния схемы

    Состояние из 3 частей: характеристики интегральной схемы, DGI представление графа и предыдещие практически one-hot заэнкоженные команды
    """
    def __init__(self, worker_index=1, reward_index=1):
        # seed_torch()
        self.is_done = None
        self.device = "cpu"
        self.pred_command = None

        self.worker_index = worker_index
        self.scheme_list = ['spi', 'ss_pcm', 'usb_phy', 'des3_area', 'fpu', 'aes_xcrypt',
                            'tinyRocket', 'pci', 'simple_spi', 'aes', 'wb_dma', 'fir',
                            'tv80', 'aes_secworks', 'dynamic_node', 'sha256', 'ac97_ctrl', 'i2c',
                            'ethernet', 'mem_ctrl', 'iir', 'sasc', 'wb_conmax', 'bp_be', 'vga_lcd', 'jpeg', 'picosoc',
                            'dft', 'idft']
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1000000,
                                            shape=(25,),
                                            dtype=np.float32)

        self.decode_action_numbers = {1: "b", 2: "rf", 3: "rw"}
        self.one_hot_action_numbers = 0.06
        self.current_one_hot = np.array([0.1, 0.1, 0.1])

        self.dict_orig = {key: [0, np.zeros(5)] for key in self.scheme_list}
        self.base_orig = {key: np.array([]) for key in self.scheme_list}

        self.basic_rewards = {key: [0, 0] for key in self.scheme_list}
        self.state = []
        self.passes_number = 0

        self.command_repeats = {key: 0 for key in self.scheme_list}
        self.reward_index = reward_index

        hidden_weights = 16
        hidden_channels = 1024

        self.model = DeepGraphInfomax(
            hidden_channels=hidden_channels, encoder=Encoder(4, hidden_channels, hidden_weights),  # (2, 1024)
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption, hidden_weights=hidden_weights)

        self.model.load_state_dict(state_dict=torch.load(PATH_TO_SAVE_DGI))
        self.model = self.model.to(self.device)

        self.ad_prev = 0

        self.graph_training_number = 0

    def reset(self, *, seed=None, options=None, graph_n=None):
        self.passes_number = 0
        self.current_one_hot = np.array([0.1, 0.1, 0.1])
        self.graph_name = self.scheme_list[self.graph_training_number]

        if self.graph_training_number < 29:
            self.graph_training_number += 1
        else:
            self.graph_training_number = 0

        if graph_n is not None:
            self.graph_name = graph_n

        get_moment_aig(self.graph_name, 0, self.worker_index)
        self.state = np.array(self.get_vector_state(), dtype=np.float64)
        if self.graph_name not in self.dict_orig:
            self.dict_orig[self.graph_name] = [0, np.zeros(5)]
            self.base_orig[self.graph_name] = np.array([])
            self.command_repeats[self.graph_name] = 0
            self.basic_rewards[self.graph_name] = [0, 0]

        if self.dict_orig[self.graph_name][0] == 0:
            self.dict_orig[self.graph_name][0] += 1
            self.dict_orig[self.graph_name][1] = self.state.copy()
            g = np.vectorize(lambda x: x + 10 ** (len(str(abs(int(x)))) - 1))

            self.base_orig[self.graph_name] = g(self.dict_orig[self.graph_name][1])

            self.dict_orig[self.graph_name][1] = np.divide(self.dict_orig[self.graph_name][1],
                                                           g(self.dict_orig[self.graph_name][1]))

            ad = get_area_delay2(self.graph_name, 0, self.worker_index)
            self.basic_rewards[self.graph_name][0], self.basic_rewards[self.graph_name][1] = ad["area"], ad["delay"]
        self.ad_prev = self.basic_rewards[self.graph_name][0] * self.basic_rewards[self.graph_name][1]

        self.state = self.dict_orig[self.graph_name][1]
        path = f"common/aig_benches/{self.graph_name}/{self.graph_name}_orig.bench"
        print(path)

        args = parseCmdLineArgs(bench=path, gml=f"common/aig_benches/{self.graph_name}/")
        setGlobalAndEnvironmentVars(args.bench, args.gml)
        networkx_graph = parseAIGBenchAndCreateNetworkXGraph()
        trainer = networkx_load(networkx_graph, device=self.device)
        dgi_graph = self.dgi_state(trainer)
        torch.seed()
        self.state = np.concatenate((self.state, dgi_graph))
        self.state = np.concatenate((self.state, self.current_one_hot))
        self.pred_command = ""
        self.is_done = False
        return self.state, {}

    def step(self, action):
        action_ = self.decode_action_numbers[action + 1]
        print(action_)
        self.current_one_hot[action] += self.one_hot_action_numbers
        get_next_scheme(self.graph_name, self.passes_number, action_, self.worker_index)

        self.passes_number += 1
        reward = self.get_reward()
        if action_ == self.pred_command:
            self.command_repeats[self.graph_name] += 1
        else:
            self.command_repeats[self.graph_name] = 0
        self.pred_command = action_

        self.state = self.get_vector_state()

        self.state = np.divide(self.state, self.base_orig[self.graph_name])

        path = f"common/aig_benches/{self.graph_name}/{self.worker_index}/{self.graph_name}_{self.passes_number}.bench"
        print("Path: ", path)
        args = parseCmdLineArgs(bench=path, gml=f"common/aig_benches/{self.graph_name}/")
        setGlobalAndEnvironmentVars(args.bench, args.gml)
        graph_networkx = parseAIGBenchAndCreateNetworkXGraph()

        trainer = networkx_load(graph_networkx, device=self.device)

        dgi_graph = self.dgi_state(trainer)
        torch.seed()
        self.state = np.concatenate((self.state, dgi_graph))
        self.state = np.concatenate((self.state, self.current_one_hot))
        self.is_done = self.is_done_()

        return self.state, reward, self.is_done, False, {}

    def get_reward(self):
        reward = 0
        passes_reward = 4

        if self.reward_index == 1 and self.passes_number == 20:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            reward_zero = (self.basic_rewards[self.graph_name][0] / area) * (
                    self.basic_rewards[self.graph_name][1] / delay) - 1
            reward = reward_zero*4

        elif self.reward_index == 2:
            if self.passes_number % passes_reward == 0:
                ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
                area, delay = ad_current["area"], ad_current["delay"]
                ad_pred = get_area_delay2(self.graph_name, self.passes_number - passes_reward, self.worker_index)

                eps_reward_curr = (self.passes_number / 20)
                reward_curr = (ad_pred["area"] / area) * (ad_pred["delay"] / delay) - 1
                reward_zero = (self.basic_rewards[self.graph_name][0] / area) * (
                            self.basic_rewards[self.graph_name][1] / delay) - 1
                reward_repeats = -(1.04 ** self.command_repeats[self.graph_name]) + 1
                reward = eps_reward_curr * reward_curr + (1 - eps_reward_curr) * reward_zero + reward_repeats
            reward *= 4

        elif self.reward_index == 3 and self.passes_number == 20:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            reward_zero = (self.basic_rewards[self.graph_name][0] / area) * (
                    self.basic_rewards[self.graph_name][1] / delay)
            if reward_zero > 1:
                reward = 1
            else:
                reward = -1

        elif self.reward_index == 4:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            reward_prev = 0.04 if self.ad_prev / (area * delay) - 1 > 0 else -0.04
            reward = reward_prev
            if self.passes_number == 20:
                reward_zero = 0.96 if (self.basic_rewards[self.graph_name][0] / area) * (
                    self.basic_rewards[self.graph_name][1] / delay) - 1 > 0 else -0.96
                reward += reward_zero
            self.ad_prev = area * delay

        elif self.reward_index == 5:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            reward_prev = (self.ad_prev / (area * delay) - 1) *0.04
            reward = reward_prev
            if self.passes_number == 20:
                reward_zero = 0.96 *((self.basic_rewards[self.graph_name][0] / area) * (
                    self.basic_rewards[self.graph_name][1] / delay) - 1)
                reward += reward_zero
            self.ad_prev = area * delay

        elif self.reward_index == 6:
            print("Testing..")

        elif self.reward_index == 8:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            reward_prev = (self.ad_prev / (area * delay) - 1)
            reward = reward_prev
            self.ad_prev = area * delay

        elif self.reward_index == 9:
            ad_current = get_area_delay2(self.graph_name, self.passes_number, self.worker_index)
            area, delay = ad_current["area"], ad_current["delay"]
            base = self.basic_rewards[self.graph_name][0] * self.basic_rewards[self.graph_name][1]
            reward_prev = self.ad_prev / base - area * delay / base
            reward = reward_prev
            self.ad_prev = area * delay
        print(reward)
        return reward

    def render(self):
        return

    def is_done_(self):
        if self.passes_number == 10:
            self.passes_number = 0
            return True
        return False

    def get_vector_state(self):
        vector_state = scheme_info(self.graph_name, self.passes_number, self.worker_index)

        return vector_state

    def dgi_state(self, test_loader):
        seed_torch()

        @torch.no_grad()
        def test():
            self.model.eval()

            zs = []
            for batch in (test_loader):
                pos_z, _, _ = self.model(batch.x, batch.edge_index, batch.batch_size)
                zs.append(pos_z.detach().cpu())
            z = torch.cat(zs, dim=0)
            return z

        s = nn.Sigmoid()
        res = test()
        res = s(res.mean(dim=0))
        res = res.numpy()
        return res
