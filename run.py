import argparse
from collections import defaultdict
import json
import random
from typing import Generator, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from agent_factory import create_agents
from env_factory import create_environment
from utils import set_seed, run_experiment, average_results


def main(config_file):
    # Load the config file
    with open(config_file, "r") as file:
        config = json.load(file)

    # set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Experiment setup
    num_experiments = 1000
    num_arms = 10

    env_generator = create_environment(config["envs"])
    agent_rewards = defaultdict(list)
    agent_opt_action = defaultdict(list)

    for env in env_generator:
        agents = list(create_agents(env.action_space, config["agents"]))
        for agent in agents:
            rewards, is_opt_action = run_experiment(env, agent)
            agent_rewards[str(agent)].append(rewards)
            agent_opt_action[str(agent)].append(is_opt_action)

    # Average the results
    average_agent_rewards = {
        agent: average_results(rewards) for agent, rewards in agent_rewards.items()
    }
    average_agent_opt_action = {
        agent: average_results(opt_action)
        for agent, opt_action in agent_opt_action.items()
    }

    # plotting
    plt.figure(figsize=(12, 8))
    for agent_name, opt_action in average_agent_opt_action.items():
        plt.plot(opt_action * 100, label=agent_name)

    # plt.title('Optimal action % of Agents in Bernoulli Multi-Armed Bandit Environments')
    plt.xlabel("Episodes")
    plt.ylabel("Optimal action %")
    plt.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments based on a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        type=str,
        required=True,
        help="Path to the configuration JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
