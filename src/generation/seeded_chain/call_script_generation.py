import argparse
import os
import uuid
from typing import Any, Dict, List

from loguru import logger

from src.config import config
from src.generation.seeded_chain.chains.call_script.generation_chains import (
    generation_resolved_chain_de_instruct,
    generation_resolved_chain_en_instruct,
    generation_unresolved_chain_de_instruct,
    generation_unresolved_chain_en_instruct,
)
from src.generation.seeded_chain.chains.call_script.validation_chains import (
    validation_chain_de_instruct,
    validation_chain_en_instruct,
)
from src.utils import (
    aggregate_json_files,
    get_random_examples,
    get_random_topics,
    load_examples,
    load_schema,
    load_topics,
    save_json,
    validate_json,
)


def main(input_path: str, output_path: str) -> None:
    """Process example calls and generate validated scripts.

    Args:
    ----
        input_path (str): The path to the input JSON file with example calls.
        output_path (str): The path to save the output JSON file.

    """
    # Load topics from the JSON file
    topics = load_topics(config.TOPICS)

    # Load example data
    examples_json = load_examples(input_path)
    random_examples = get_random_examples(examples_json, config.NUM_EXAMPLE_SAMPLES)
    logger.info("Loaded example data.")
    logger.info(f"Creating with model: {config.MODEL_NAME}")

    # Load schema for example data
    schema = load_schema(config.SCHEMA_SCRIPT)
    logger.info("Loaded JSON schema.")

    # Fetch topics for the generation prompt
    if config.FIXED_TOPICS:
        topics = [
            "Erneuerung Kreditkarte",
            "Adressänderung",
            "Börsenauftrag",
            "Hypotheken",
            "Fremdwährungen bestellen",
        ]
        logger.info(f"Using fixed topics: {topics}")
    else:
        topics = get_random_topics(topics, config.NUM_TOPIC_SAMPLES)
        logger.info(f"Using random topics: {topics}")

    # Create list to store the new examples
    new_calls: List[Dict[str, Any]] = []

    if config.INSTRUCT_LANG == "de":
        generation_resolved_chain = generation_resolved_chain_de_instruct
        generation_unresolved_chain = generation_unresolved_chain_de_instruct
        validation_chain = validation_chain_de_instruct
    else:
        generation_resolved_chain = generation_resolved_chain_en_instruct
        generation_unresolved_chain = generation_unresolved_chain_en_instruct
        validation_chain = validation_chain_en_instruct

    for topic in topics:
        logger.info(f"Processing topic: {topic}")
        for example in random_examples:
            try:
                logger.info(f"Processing example: {example}")
                generated_resolved_script = generation_resolved_chain.invoke(
                    {
                        "structure_json": schema,
                        "example_json": example,
                        "topic": topic,
                    }
                )
                logger.debug(f"Generated resolved script: {generated_resolved_script}")

                generated_unresolved_script = generation_unresolved_chain.invoke(
                    {
                        "structure_json": schema,
                        "example_json": example,
                        "topic": topic,
                    }
                )
                logger.debug(
                    f"Generated unresolved script: {generated_unresolved_script}"
                )

                validated_resolved_script = validation_chain.invoke(
                    {"script_json": generated_resolved_script}
                )
                logger.debug(f"Validated script: {validated_resolved_script}")

                validated_unresolved_script = validation_chain.invoke(
                    {"script_json": generated_unresolved_script}
                )
                logger.debug(f"Validated script: {validated_unresolved_script}")

                # Add metadata to the validated scripts
                validated_resolved_script["call_id"] = str(uuid.uuid4())
                validated_resolved_script["model"] = config.MODEL_NAME
                validated_resolved_script["examples"] = input_path.split("/")[-1]
                validated_resolved_script["topic"] = topic
                validated_resolved_script["resolved"] = True
                validated_resolved_script["instruct_lang"] = config.INSTRUCT_LANG

                validated_unresolved_script["call_id"] = str(uuid.uuid4())
                validated_unresolved_script["model"] = config.MODEL_NAME
                validated_unresolved_script["examples"] = input_path.split("/")[-1]
                validated_unresolved_script["topic"] = topic
                validated_unresolved_script["resolved"] = False
                validated_unresolved_script["instruct_lang"] = config.INSTRUCT_LANG

                # Validate the example script against the schema
                if validate_json(validated_resolved_script, schema):
                    logger.info("Validation successful for resolved script.")
                    new_calls.append(validated_resolved_script)
                else:
                    logger.warning("Validation failed for resolved script, skipping.")

                if validate_json(validated_unresolved_script, schema):
                    logger.info("Validation successful for unresolved script.")
                    new_calls.append(validated_unresolved_script)
                else:
                    logger.warning("Validation failed for unresolved script, skipping.")

            except Exception as e:
                logger.error(f"Error processing example: {e}")
                continue

    # Save the final validated scripts to a JSON file
    output_json = {"calls": new_calls}

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(output_json, output_path)

    logger.info(f"Processing complete. Validated call scripts saved to {output_path}")

    aggregate_json_files(config.OUTPUT_FOLDER, config.AGGREGATED_JSON)

    logger.info(f"Aggregated JSON files saved to {config.AGGREGATED_JSON}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Process example calls and generate validated scripts."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file with example calls.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/validated_call_scripts.json",
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--num_topic_samples",
        type=int,
        default=10,
        required=False,
        help="Number of topics that samples should be generated.",
    )
    parser.add_argument(
        "--num_example_samples",
        type=int,
        default=10,
        required=False,
        help="Number of examples that samples should be generated.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--fixed_topics",
        type=bool,
        default=False,
        help="Use fixed topics for generation.",
    )
    parser.add_argument(
        "--instruct_lang",
        type=str,
        default="en",
        help="The language of the instructions.",
    )
    args = parser.parse_args()

    main(args.input, args.output)
