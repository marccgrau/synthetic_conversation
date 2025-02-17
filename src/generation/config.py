from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for dialogue summarization."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", cli_parse_args=True
    )

    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    ANTHROPIC_API_KEY: str
    NVIDIA_API_KEY: str
    HF_TOKEN: str
    DEEPL_API_KEY: str
    TOGETHER_AI_API_KEY: str
    MODEL_NAME: str = "gpt-4o-mini"
    TOPICS: str = "seed_data/topics.json"
    SCHEMA_SCRIPT: str = "src/schemas/call_script.json"
    INPUT: str = "seed_data/dialogsum_final_eval.json"
    OUTPUT: str = "generated_data/generated_calls.json"
    NUM_TOPIC_SAMPLES: int = 10
    NUM_EXAMPLE_SAMPLES: int = 10
    OUTPUT_FOLDER: str = "output"
    AGGREGATED_JSON: str = "synthetic_data/aggregated_data.json"
    MAX_TOKENS: int = 4096
    FIXED_TOPICS: bool = False
    EVAL_DATA: str = "synthetic_data/eval_data.json"
    INSTRUCT_LANG: str = "en"
    GROQ_API_KEY: str
    FIREWORKS_API_KEY: str
    TOOLHOUSE_API_KEY: str


config = Config()
