import numpy as np
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch


def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def save_results(results_list, path_log):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    results_list.insert(0, dt_string)
    with open(path_log, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(results_list)