from langchain_core.output_parsers import JsonOutputParser

from src.generation.seeded_chain.models import get_llm
from src.generation.seeded_chain.prompts.validation_de_instruct import (
    validation_template as validation_template_de_instruct,
)
from src.generation.seeded_chain.prompts.validation_en_instruct import (
    validation_template as validation_template_en_instruct,
)

llm = get_llm()
parser = JsonOutputParser()

validation_chain_en_instruct = validation_template_en_instruct | llm | parser

validation_chain_de_instruct = validation_template_de_instruct | llm | parser
