import json
import os
import random
from typing import Any, Dict, List

import jsonschema
from datasets import load_dataset
from jsonschema import validate
from loguru import logger


def load_examples(file_path_or_dataset: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load examples from a JSON file or Hugging Face dataset and return them in a standardized format.

    Args:
    ----
        file_path_or_dataset (str): The path to the JSON file or the Hugging Face dataset identifier.

    Returns:
    -------
        Dict[str, List[Dict[str, Any]]]: The loaded and standardized examples in the required format.

    """
    # Check if the input is a local file
    if os.path.isfile(file_path_or_dataset):
        logger.debug(f"Attempting to load JSON from file: {file_path_or_dataset}")
        try:
            with open(file_path_or_dataset, "r") as file:
                data = json.load(file)
                logger.debug(
                    f"Successfully loaded JSON from file: {file_path_or_dataset}"
                )
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path_or_dataset}")
            raise FileNotFoundError(f"File not found: {file_path_or_dataset}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path_or_dataset}: {e}")
            raise ValueError(f"Error decoding JSON from {file_path_or_dataset}: {e}")

    # Check if the input is a Hugging Face dataset identifier
    try:
        logger.debug(
            f"Attempting to load dataset from Hugging Face: {file_path_or_dataset}"
        )
        dataset = load_dataset(file_path_or_dataset)
        data = {"calls": []}

        # Convert dataset splits to a consistent format
        for split in dataset.keys():
            # Convert each record in the dataset split to a dictionary matching the expected format
            for record in dataset[split]:
                call_entry = {
                    "call_id": record["call_id"],
                    "script": record["script"],
                    "callback_note": record["callback_note"],
                    "model": record.get("model", "unknown_model"),
                    "examples": record.get("examples", "example_calls.json"),
                    "topic": record.get("topic", "unknown_topic"),
                    "instruct_lang": record.get("instruct_lang", "de"),
                    "generation_method": record.get(
                        "generation_method", "unknown_method"
                    ),
                }
                data["calls"].append(call_entry)

        logger.debug(
            f"Successfully loaded dataset from Hugging Face: {file_path_or_dataset}"
        )
        return data

    except Exception as e:
        logger.error(f"Error loading dataset from Hugging Face: {e}")
        raise ValueError(f"Error loading dataset from Hugging Face: {e}")


def save_json(data: Any, file_path: str) -> None:
    """Save JSON data to a file.

    Args:
    ----
        data (Any): The JSON data to save.
        file_path (str): The path to the file where the data should be saved.

    """
    logger.debug(f"Attempting to save JSON to file: {file_path}")
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
            logger.debug(f"Successfully saved JSON to file: {file_path}")
    except IOError as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise IOError(f"Error saving JSON to {file_path}: {e}")


def load_schema(schema_path: str) -> Dict[str, Any]:
    """Load the JSON schema from a file.

    Args:
    ----
        schema_path (str): The path to the JSON schema file.

    Returns:
    -------
        Dict[str, Any]: The JSON schema.

    """
    with open(schema_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_json(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON data against a given schema.

    Args:
    ----
        data (Dict[str, Any]): The JSON data to be validated.
        schema (Dict[str, Any]): The schema to validate the JSON data against.

    Returns:
    -------
        bool: True if the JSON data is valid according to the schema, False otherwise.

    """
    logger.debug("Validating JSON data against schema")
    try:
        validate(instance=data, schema=schema)
        logger.info("JSON data is valid")
        return True
    except jsonschema.exceptions.ValidationError as err:
        logger.error(f"JSON data is invalid: {err.message}")
        return False


def load_topics(file_path: str) -> List[str]:
    """Load topics from a JSON file.

    Args:
    ----
        file_path (str): The path to the JSON file containing the topics.

    Returns:
    -------
        List[str]: A list of topics.

    """
    logger.debug(f"Loading topics from file: {file_path}")
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            topics = data.get("topics", [])
            logger.debug(f"Successfully loaded topics from file: {file_path}")
            return topics
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise ValueError(f"Error decoding JSON from {file_path}: {e}")


def get_random_topics(topics: List[str], num_topics: int) -> List[str]:
    """Select a random topic from the list of topics.

    Args:
    ----
        topics (List[str]): The list of topics to choose from.
        num_topics (int): The number of topics to select.

    Returns:
    -------
        str: A randomly selected topic.

    """
    logger.debug("Selecting a random topic from the list")
    if not topics:
        logger.error("The list of topics is empty")
        raise ValueError("The list of topics is empty.")
    topics = random.sample(topics, num_topics)
    logger.debug(f"Selected random topics: {topics}")
    return topics


def get_random_examples(
    examples: Dict[str, List[Dict[str, Any]]], num_examples: int
) -> List[Dict[str, Any]]:
    """Select random examples from the list of examples.

    Args:
    ----
        examples (Dict[str, List[Dict[str, Any]]]): The examples to choose from.
        num_examples (int): The number of examples to select.

    Returns:
    -------
        List[Dict[str, Any]]: A list of randomly selected examples.

    """
    logger.debug("Selecting random examples from the list")
    if not examples:
        logger.error("The list of examples is empty")
        raise ValueError("The list of examples is empty.")

    # Get the list of calls from the dictionary
    call_list: List[Dict[str, Any]] = examples.get("calls", [])

    # Check if there are enough examples to select
    if len(call_list) == 0:
        logger.error("No calls available in the list to select from.")
        raise ValueError("No calls available in the list to select from.")

    if num_examples > len(call_list):
        logger.warning(
            f"Requested {num_examples} examples, but only {len(call_list)} available. Adjusting to {len(call_list)}."
        )
        num_examples = len(call_list)

    # Ensure the number of examples is not negative
    if num_examples < 0:
        logger.error("Number of examples to select cannot be negative.")
        raise ValueError("Number of examples to select cannot be negative.")

    # Select random examples from the list of calls
    random_examples: List[Dict[str, Any]] = random.sample(call_list, num_examples)

    logger.debug(f"Selected random examples: {random_examples}")
    return random_examples


def aggregate_json_files(input_folder: str, output_file: str) -> None:
    """Aggregate JSON files from a specified input folder into a single JSON file, replacing the call_id with a unique ID.

    Args:
    ----
        input_folder (str): The path to the folder containing the input JSON files.
        output_file (str): The path to the output JSON file where the aggregated data will be saved.

    Returns:
    -------
        None

    """
    # Initialize an empty list to store the aggregated data
    aggregated_data: List[Dict[str, Any]] = []

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)
            # Read and parse the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for call in data["calls"]:
                    # Append the data to the aggregated list
                    aggregated_data.append(call)

    # Prepare the final aggregated structure
    final_data: Dict[str, List[Dict[str, Any]]] = {"calls": aggregated_data}

    # Write the aggregated data to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(final_data, file, ensure_ascii=False, indent=4)
