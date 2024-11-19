from langchain_core.output_parsers import JsonOutputParser

from src.generation.seeded_chain.models import get_llm
from src.generation.seeded_chain.prompts.generation_resolved_de_instruct import (
    generation_resolved_template as generation_resolved_template_de_instruct,
)
from src.generation.seeded_chain.prompts.generation_resolved_en_instruct import (
    generation_resolved_template as generation_resolved_template_en_instruct,
)
from src.generation.seeded_chain.prompts.generation_unresolved_de_instruct import (
    generation_unresolved_template as generation_unresolved_template_de_instruct,
)
from src.generation.seeded_chain.prompts.generation_unresolved_en_instruct import (
    generation_unresolved_template as generation_unresolved_template_en_instruct,
)

llm = get_llm()
parser = JsonOutputParser()

generation_resolved_chain_en_instruct = (
    generation_resolved_template_en_instruct | llm | parser
)

generation_unresolved_chain_en_instruct = (
    generation_unresolved_template_en_instruct | llm | parser
)

generation_resolved_chain_de_instruct = (
    generation_resolved_template_de_instruct | llm | parser
)

generation_unresolved_chain_de_instruct = (
    generation_unresolved_template_de_instruct | llm | parser
)
