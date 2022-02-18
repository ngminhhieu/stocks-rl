import gym

from stable_baselines3 import PPO
from framework.env import Environment
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

# Parallel environments
def trainPPO(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("test")

    del model # remove to demonstrate saving and loading

with open("data/config.yaml") as f:
    config = yaml.safe_load(f)
env = DummyVecEnv([lambda: Environment(config)])
# trainPPO(env)

model = PPO.load("test")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
print(rewards)