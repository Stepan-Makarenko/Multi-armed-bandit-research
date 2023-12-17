from typing import Dict, Union

import numpy as np
from options import (
    parse_list_option,
    parse_random_list_option,
    parse_repeated_list_option,
    parse_float_option,
    parse_int_option,
)

OPTION_FACTORY = {
    "list": parse_list_option,
    "repeated_list": parse_repeated_list_option,
    "random_list": parse_random_list_option,
    "float": parse_float_option,
    "int": parse_int_option
    # Add other environment classes here
}


def create_option(option_config: Dict) -> Union[np.ndarray, float]:
    """
    Factory function to handle different types of options.

    Args:
    option_config (dict): The option config

    Returns:
    Parsed option value in the correct type.
    """
    option_class_name = option_config.get("class")
    option_func = OPTION_FACTORY.get(option_class_name)
    assert (
        option_func is not None
    ), "Option class with name {option_class_name} does not exist"
    return option_func(option_config)
