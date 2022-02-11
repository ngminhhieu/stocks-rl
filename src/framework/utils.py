import numpy as np
from datetime import datetime
import csv
import networkx as nx
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import torch


def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def plot(results, prob, num_gt, results_dir):
    results = np.array(results)
    results = pd.DataFrame(
        results, columns=['episode', 'reward', 'agent_loss', 'time'])
    results.to_csv(results_dir + "/episodes.csv", index=False)
    prob_df = pd.DataFrame.from_dict(prob)
    prob_df.to_csv(results_dir + "/prob.csv", index=False)

    # Visualization
    # sns.lineplot(data=results.reward/num_gt, color="g")
    plt.legend(labels=["Reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(results_dir + '/rewards.png')


def save_results(results_list, path_log):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    results_list.insert(0, dt_string)
    with open(path_log, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(results_list)