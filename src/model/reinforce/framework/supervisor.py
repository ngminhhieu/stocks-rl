import os
import numpy as np
import torch.optim as optim
from utils.env import Environment
from utils.model import save_results
from model.reinforce.framework.memory import Memory
from model.reinforce.framework.agent import Agent
import torch
from torch.distributions import Categorical
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class ReinforceSupervisor():
    def __init__(self, config):
        self._config = config
        self._run_name = config["project"]["run_name"]
        self._project_name = config["project"]["project_name"]

        self._episode = config["model"]["episode"]
        self._gamma = config["model"]["gamma"]
        self._lr = config["model"]["learning_rate"]

        # log folder
        self._pretrained_log = "log/{}/{}".format(self._project_name, self._run_name)
        if not os.path.exists(self._pretrained_log):
            os.makedirs(self._pretrained_log)
        
        # Initialize model
        
        self._env = Environment(config)
        self._X_train = self._env._X_train
        self._y_train = self._env._y_train
        self._X_test = self._env._X_test
        self._y_test = self._env._y_test
        self._model = Agent(self._env._get_num_fts(), 32, 16).cuda()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._memory = Memory()


    def train(self):
        results = {}
        results["accuracy"] = []
        results["training"] = []  # results of each episode
        results["prob"] = {}  # to check probability of each pair after one episode
        results["policy"] = []

        print("Training...")
        for ep in tqdm(range(1, self._episode + 1)):
            # Reset environment
            state, ep_reward = self._env.reset(), 0

            # Get policy, action and reward
            while True:
                action, policy = self._get_action(state)
                if ep == self._episode:
                    results["policy"].append(policy)
                    
                next_state, reward, done, _ = self._env.step(action)

                # add reward
                self._memory._reward.append(reward)
                # ep_reward += reward
                ep_reward = reward
                if done:
                    break

                # next state
                state = next_state

            # Train model
            loss = self._finish_episode()
            if ep % 10 == 0:
                print("Episode: {}   Reward: {}   Agent loss: {}".format(ep, ep_reward, loss.cpu().detach().numpy()))

        torch.save(self._model.state_dict(), os.path.join(self._pretrained_log, "best.pt"))

        return results, self._model


    def test(self):
        self._model.load_state_dict(torch.load(os.path.join(self._pretrained_log, "best.pt")))
        self._model.cuda()
        self._model.eval()

        print("Testing...")
        training_total_match = 0
        testing_total_match = 0
        training_acc = 0
        testing_acc = 0

        for index in tqdm(range(len(self._X_train)), desc="Evaluate training accuracy"):
            action, p = self._get_action(self._X_train[index])
            if action == self._y_train[index][-1]:
                training_total_match += 1

        for index in tqdm(range(len(self._X_test)), desc="Evaluate training accuracy"):
            action, p = self._get_action(self._X_test[index])
            print(p)
            if action == self._y_test[index][-1]:
                testing_total_match += 1

        print("Saving results...")
        training_acc = training_total_match/len(self._X_train)
        print("Training accuracy: ", training_acc)
        testing_acc = testing_total_match/len(self._X_test)
        print("Testing accuracy: ", testing_acc)
        save_results([training_acc, testing_acc], self._pretrained_log + "/accuracy.csv")

        # obs = self._env.reset()
        # while True:
        #     action, _states = self._model.predict(obs)
        #     obs, rewards, done, info = self._env.step(action)
        #     if done:
        #         break
        # print(rewards)

    
    def _finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self._memory._reward[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(np.array(returns)).cuda()
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self._memory._saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
        del self._memory._reward[:]
        del self._memory._saved_log_probs[:]
        return policy_loss


    def _get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).cuda()
        policy = self._model(state)
        m = Categorical(policy)
        action = m.sample()
        self._memory._saved_log_probs.append(m.log_prob(action))
        return action.item(), policy.cpu().data.numpy()