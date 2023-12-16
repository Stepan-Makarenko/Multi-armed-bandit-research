from typing import Dict, Generator, List
from gymnasium import Space
from agents import RandomAgent, ExploreAndGreed, UCB1
from option_factory import create_option

# Agent factory map
AGENT_FACTORY = {
    "RandomAgent": RandomAgent,
    "ExploreAndGreed": ExploreAndGreed,
    "UCB1": UCB1,
    # Add other agent classes here
}


def create_agents(action_space: Space, agent_configs: List[Dict]) -> Generator:
    """
    Generator function to create agent instances based on the list of configurations.
    """
    for agent_config in agent_configs:
        agent_class_name = agent_config.get("class")
        agent_class = AGENT_FACTORY.get(agent_class_name)
        assert agent_class is not None, f"Class {agent_class_name} does not excist"

        kwargs = {"action_space": action_space}
        for name, value in agent_config.get("options", {}).items():
            kwargs[name] = create_option(value)

        yield agent_class(**kwargs)
