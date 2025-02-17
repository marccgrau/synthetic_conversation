import os

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def configure_llm_settings(model_name: str, embedding_model: str) -> None:
    """Configure LLM and embedding model settings for LlamaIndex.

    Parameters
    ----------
    model_name : str
        The name of the language model to be used (e.g., "gpt-4o-mini").
    embedding_model : str
        The name of the embedding model to be used (e.g., "text-embedding-3-large").

    Returns
    -------
    None

    """
    # Initialize the LLM with the provided model name
    llm: OpenAI = OpenAI(
        model=model_name,
        temperature=0.2,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Initialize the embedding model with the specified embedding model name
    embed_model: OpenAIEmbedding = OpenAIEmbedding(
        model=embedding_model,
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    conversable_agent_llm = {
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_type": "openai",
    }

    # Set the LLM and embedding model in the global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.conversable_agent_llm = conversable_agent_llm
