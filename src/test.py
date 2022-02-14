import torch
import os
from tqdm import tqdm
from torch.distributions import Categorical
from framework.env import Environment
from framework.agent import Agent
from framework.utils import save_results
import yaml


def get_log_dir(run_name):
    results_dir = "./log/test/results/{}".format(run_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    policy = agent(state)
    m = Categorical(policy)
    action = m.sample()
    return action.item(), policy.cpu().data.numpy()


if __name__ == '__main__':
    
    with open("data/config.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["target_col"] = "trend"
    config["data"]["indicators"]["trend"] = {"trend_up_threshold": 0, "trend_down_threshold": 0}

    print("Beginning the training process...")
    results_dir = get_log_dir(config["project"]["run_name"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Building the environment...")
    env = Environment(config)

    print("Intitializing agent...")
    agent = Agent(env._get_num_fts(), 32, 16)
    agent.load_state_dict(torch.load("./log/train/weight/{}/{}".format(config["project"]["run_name"], "final.pt")))
    agent.to(device)
    agent.eval()

    print("Testing...")
    training_total_match = 0
    testing_total_match = 0
    training_acc = 0
    testing_acc = 0

    X_train = env._X_train
    y_train = env._y_train
    X_test = env._X_test
    y_test = env._y_test
    for index in tqdm(range(len(X_train)), desc="Evaluate training accuracy"):
        action, p = get_action(X_train[index])
        if action == y_train[index][-1]:
            training_total_match += 1

    for index in tqdm(range(len(X_test)), desc="Evaluate training accuracy"):
        action, p = get_action(X_test[index])
        print(p)
        if action == y_test[index][-1]:
            testing_total_match += 1

    print("Saving results...")
    training_acc = training_total_match/len(X_train)
    print("Training accuracy: ", training_acc)
    testing_acc = testing_total_match/len(X_test)
    print("Testing accuracy: ", testing_acc)
    save_results([training_acc, testing_acc], results_dir + "/accuracy.csv")
