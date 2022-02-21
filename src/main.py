import os
import sys
import argparse
import yaml
import numpy as np
import random
import torch
from model.ppo.framework.supervisor import PPOSupervisor
from model.reinforce.framework.supervisor import ReinforceSupervisor

def seed(number):
    random.seed(number)
    np.random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)


def run_model(args, config):
    model = None
    if args.model == "ppo":
        model = PPOSupervisor(config)
    elif args.model == "reinforce":
        model = ReinforceSupervisor(config)
            
    if args.train:
        model.train()

    if args.test:
        model.test()

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default="reinforce",
                        type=str)
    parser.add_argument('--train',
                        default=0,
                        type=int,
                        help='Train the model')
    parser.add_argument('--test',
                        default=1,
                        type=int,
                        help='Test the model')
    parser.add_argument('--config',
                        default="model/reinforce/config/test.yaml",
                        type=str)
    parser.add_argument('--seed',
                        default=42,
                        type=int)
    
    # Read config
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed(args.seed)
    run_model(args, config)