import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(self.log_dir, "best_model/")
        self.best_mean_reward = -np.inf
        self.verbose = verbose
        self._learning_rate_start = None
        self.reward_checker = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        x, y = [0], [0]
        if self.n_calls % 20 == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            self.tb_formatter.writer.add_scalar("rewards/",
                                                y[-1],
                                                self.n_calls)
            self.logger.record("rollout/reward_value", self.locals["rewards"])
        if self.n_calls % self.check_freq == 0 and self.n_calls >= 20:
            if len(x) > 0:
                mean_reward = np.mean(y[-12:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    if mean_reward > self.best_mean_reward or np.mean(y[-8:] > 0) == 1:
                        if self.best_mean_reward < mean_reward:
                            self.best_mean_reward = mean_reward

                        if self.verbose >= 1:
                            print(f"Saving new best model to {self.save_path}")
                            self.model.save(self.save_path + str(self.n_calls) + "_" + str(mean_reward) + ".zip")
        return True

    def _on_rollout_end(self) -> None:
        ...
