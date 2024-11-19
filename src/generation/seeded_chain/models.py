from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import config


def get_llm():
    """Return an instance of a language model based on the provided model name in the config.

    Returns
    -------
        An instance of a language model.

    Raises
    ------
        ValueError: If the provided model name is not supported.

    """
    model_name = config.MODEL_NAME.lower()

    model_map = {
        "gpt": {
            "class": ChatOpenAI,
            "params": {
                "model_name": model_name,
                "openai_api_key": config.OPENAI_API_KEY,
            },
        },
        "gemini": {
            "class": ChatVertexAI,
            "params": {
                "model": model_name,
                "convert_system_message_to_human": True,
                "max_tokens": config.MAX_TOKENS,
            },
        },
        "claude": {
            "class": ChatAnthropic,
            "params": {
                "model": model_name,
                "api_key": config.ANTHROPIC_API_KEY,
                "max_tokens": config.MAX_TOKENS,
            },
        },
        "sonnet": {
            "class": ChatAnthropic,
            "params": {
                "model": model_name,
                "api_key": config.ANTHROPIC_API_KEY,
                "max_tokens": config.MAX_TOKENS,
            },
        },
        "haiku": {
            "class": ChatAnthropic,
            "params": {
                "model": model_name,
                "api_key": config.ANTHROPIC_API_KEY,
                "max_tokens": config.MAX_TOKENS,
            },
        },
        "nvidia": {
            "class": ChatNVIDIA,
            "params": {"model": model_name, "api_key": config.NVIDIA_API_KEY},
        },
        "fireworks": {
            "class": ChatFireworks,
            "params": {"model": model_name, "api_key": config.FIREWORKS_API_KEY},
        },
        "llama": {
            "class": ChatGroq,
            "params": {"model": model_name, "api_key": config.GROQ_API_KEY},
        },
        "gemma": {
            "class": ChatGroq,
            "params": {"model": model_name, "api_key": config.GROQ_API_KEY},
        },
    }

    for key in model_map:
        if key in model_name:
            logger.info(f"Returning: {key}")
            model_class = model_map[key]["class"]
            model_params = model_map[key]["params"]
            logger.info(f"Returning: {model_name}")
            return model_class(**model_params)

    raise ValueError(f"Model {config.MODEL_NAME} not supported.")
