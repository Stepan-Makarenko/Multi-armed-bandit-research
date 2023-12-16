from typing import Dict, List, Generator
from envs import MultiArmedBernoulliBanditEnv, MultiArmedGausianBanditEnv
from option_factory import create_option


# Environment factory maps
ENV_FACTORY = {
    "MultiArmedBernoulliBanditEnv": MultiArmedBernoulliBanditEnv,
    "MultiArmedGausianBanditEnv": MultiArmedGausianBanditEnv
    # Add other environment classes here
}


def create_environment(env_configs: List[Dict]) -> Generator:
    """
    Generator function to create environment instances based on the list of configurations.
    """
    for env_config in env_configs:
        env_class_name = env_config.get("class")
        env_class = ENV_FACTORY.get(env_class_name)
        assert env_class is not None, f"Class {env_class_name} does not excist"

        repeat = env_config.get("repeat", 1)
        for _ in range(repeat):
            kwargs = {}
            for name, value in env_config.get("options", {}).items():
                kwargs[name] = create_option(value)

            yield env_class(**kwargs)
