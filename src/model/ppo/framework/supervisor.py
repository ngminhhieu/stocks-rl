import os
from utils.env import Environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class PPOSupervisor():
    def __init__(self, config):
        self._config = config
        self._run_name = config["project"]["run_name"]
        self._project_name = config["project"]["project_name"]

        # log folder
        self._pretrained_log = "log/{}/{}".format(self._project_name, self._run_name)
        if not os.path.exists(self._pretrained_log):
            os.makedirs(self._pretrained_log)
        
        # Initialize model
        self._env = DummyVecEnv([lambda: Environment(config)])
        self._model = PPO("MlpPolicy", self._env, verbose=1)

    def train(self):
        self._model.learn(total_timesteps=25000)
        self._model.save(os.path.join(self._pretrained_log, "weights"))


    def test(self):
        self._model.load(os.path.join(self._pretrained_log, "weights"))
        obs = self._env.reset()
        while True:
            action, _states = self._model.predict(obs)
            obs, rewards, done, info = self._env.step(action)
            if done:
                break
        print(rewards)