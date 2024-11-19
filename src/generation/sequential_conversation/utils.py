from typing import Any, Dict, List, Tuple, Union

import yaml
from groq import Groq
from openai import OpenAI as OpenAIClient


def termination_msg(x: Dict[str, Any]) -> bool:
    """Check if the given input is a termination message.

    Parameters
    ----------
    x : Dict[str, Any]
        The convo to be checked.

    Returns
    -------
    bool
        True if the input is a termination message, False otherwise.

    """
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


def load_yaml(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """Load and parse a YAML file.

    Parameters
    ----------
    file_path : str
        The path to the YAML file to be loaded.

    Returns
    -------
    Union[Dict[str, Any], List[Any]]
        The parsed contents of the YAML file, which can be a dictionary or a list.

    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_client(config_list: List[Dict[str, Any]], tags: List[str]) -> Tuple[Union[OpenAIClient, Groq], Dict[str, Any]]:
    """Fetch client and config based on provided tags.

    Parameters
    ----------
    config_list : List[Dict[str, Any]]
        A list of configuration dictionaries containing client settings.

    tags : List[str]
        A list of tags to filter and identify the correct configuration.

    Returns
    -------
    Tuple[Union[OpenAIClient, Groq], Dict[str, Any]]
        A tuple containing the initialized client and the corresponding configuration.

    Raises
    ------
    ValueError
        If no matching configuration is found or if the provider is unknown.

    """
    config = next((cfg for cfg in config_list if all(tag in cfg["tags"] for tag in tags)), None)
    if config:
        if "openai" in tags:
            client = OpenAIClient(api_key=config["api_key"])
        elif "groq" in tags:
            client = Groq(api_key=config["api_key"])
        else:
            raise ValueError("Unknown provider")
        return client, config
    else:
        raise ValueError("No matching configuration found.")
