import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from framework.utils import seed
from framework.env import Environment
from framework.agent import Agent
import time
import sys
import argparse
import os
import wandb
from framework.memory import Memory
import yaml


def get_log_dir(run_name):
    results_dir = "./log/train/results/{}".format(run_name)
    weight_dir = "./log/train/weight/{}".format(run_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    return results_dir, weight_dir


def get_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    policy = agent(state)
    m = Categorical(policy)
    action = m.sample()
    memory._saved_log_probs.append(m.log_prob(action))
    return action.item(), policy.cpu().data.numpy()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in memory._reward[::-1]:
        R = r + config["model"]["gamma"] * R
        returns.insert(0, R)
    returns = torch.tensor(np.array(returns)).to(device)
    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(memory._saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del memory._reward[:]
    del memory._saved_log_probs[:]
    return policy_loss

def train():
    early_stopping = 0
    results = {}
    num_gt = len(env._list_state)
    results["accuracy"] = []
    results["training"] = []  # results of each episode
    results["prob"] = {}  # to check probability of each pair after one episode
    results["policy"] = []

    print("Training...")
    for ep in tqdm(range(1, config["model"]["episode"] + 1)):
        start_episode = time.time()

        # Reset environment
        state, ep_reward = env.reset(), 0
        total_reward = num_gt * config["model"]["tp_score"] + \
            (len(env._list_state) - num_gt) * config["model"]["tn_score"]
        # Get policy, action and reward
        while True:
            action, policy = get_action(state)
            if ep == config["model"]["episode"]:
                results["policy"].append(policy)
                
            next_state, reward, done = env.step(action)

            # add reward
            memory._reward.append(reward)
            ep_reward += reward
            if done:
                break

            # next state
            state = next_state

        # wandb.log({"reward": ep_reward})

        # Train model
        loss = finish_episode()
        end_episode = time.time()

        results["training"].append(
            [ep, ep_reward, loss.cpu().detach().numpy(), end_episode - start_episode])
        results["accuracy"].append(
            [env._matched["tp"], env._matched["tn"], env._matched["fp"], env._matched["fn"]])
        torch.save(agent.state_dict(), weight_dir + "/best.pt")
        # Monitoring
        if ep % 10 == 0:
            print("Episode: {}   Reward: {}/{}   Agent loss: {}".format(ep,
                  ep_reward, total_reward, loss.cpu().detach().numpy()))
        if ep % 10 == 0:
            print(env._matched)

        # Early stopping
        if ep_reward == total_reward:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping == config["model"]["earlystop"]:
            print("Early stopping")
            print("Goal reached! Reward: {}/{}".format(ep_reward, total_reward))
            break

    return results, agent


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='Seed')

    # Read config and args
    args = parser.parse_args()
    seed(args)
    with open("data/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Beginning the training process...")
    results_dir, weight_dir = get_log_dir(config["project"]["run_name"])
    # run = wandb.init(project=config["project"]["project_name"], entity="bkai", config=args, resume=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Building the environment...")
    env = Environment(config)
    memory = Memory()

    print("Intitializing agent...")
    agent = Agent(env._get_num_fts(), 32, 16)

    optimizer = optim.Adam(agent.parameters(), lr=config["model"]["learning_rate"])
    agent.to(device)
    agent.train()
    results, agent = train()
    torch.save(agent.state_dict(), os.path.join(weight_dir, "final.pt"))