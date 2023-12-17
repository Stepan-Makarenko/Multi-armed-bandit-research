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


def main(config_file, plot_prefix):
    # Load the config file
    with open(config_file, "r") as file:
        config = json.load(file)

    # set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Experiment setup
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
    # average reward
    plt.figure(figsize=(12, 8))
    for agent_name, rewards in average_agent_rewards.items():
        plt.plot(rewards, label=agent_name)

    # plt.title('Average Rewards of Agents in Bernoulli Multi-Armed Bandit Environments')
    fontsize = 20
    plt.xlabel("Episodes", fontsize=fontsize)
    plt.ylabel("Average Reward", fontsize=fontsize)
    plt.legend()
    # Adding a grid
    plt.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)
    # plt.show()
    save_path = plot_prefix + "_avg_reward"
    plt.savefig(save_path)

    # % of optimal actions
    plt.figure(figsize=(12, 8))
    for agent_name, opt_action in average_agent_opt_action.items():
        plt.plot(opt_action * 100, label=agent_name)

    # plt.title('Optimal action % of Agents in Bernoulli Multi-Armed Bandit Environments')
    plt.xlabel("Episodes", fontsize=fontsize)
    plt.ylabel("Optimal action %", fontsize=fontsize)
    plt.legend()
    # Adding a grid
    plt.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)
    # plt.show()
    save_path = plot_prefix + "_opt_action"
    plt.savefig(save_path)


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
    parser.add_argument(
        "-pp",
        "--plot_prefix",
        dest="plot_prefix",
        default="./example_plot",
        type=str,
        required=False,
        help="Path to store resulting plots",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file, args.plot_prefix)
