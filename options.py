from typing import Dict, List
import numpy as np

DEFAULT_FLOAT_VALUE = 0


def parse_list_option(option_config: Dict) -> np.ndarray:
    option_value = option_config.get("value", [])
    return np.array(option_value)


def parse_repeated_list_option(option_config: Dict) -> np.ndarray:
    option_value = option_config.get("value", [])
    times = option_config.get("times", 1)
    return np.array(option_value * times)


def parse_random_list_option(option_config: Dict) -> np.ndarray:
    size = option_config.get("size", 1)
    return np.random.randn(size)  # This is a placeholder


def parse_float_option(option_config: Dict) -> float:
    option_value = option_config.get("value", DEFAULT_FLOAT_VALUE)
    return option_value
