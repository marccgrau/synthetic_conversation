"""
Script for Running Conversation Simulations Between Service and Customer Agents

This script simulates interactions between AI-driven customer and service agents using different configurations, models, and scenarios.
It utilizes various AI agent types (RAG, Society of Mind, or Simple Service Agent) and executes conversational flows based on predefined scenarios.

Key Features:
- Allows configuration of different LLM models and providers.
- Supports multiple iterations of conversation simulations.
- Implements different agent types to model diverse conversational behaviors.
- Loads PDF and Web-based knowledge indexes for improved AI response generation.
- Saves each simulated conversation with a unique identifier.

Modules:
- `argparse`: Handles command-line argument parsing.
- `json`, `os`, `uuid`, `datetime`: Manages result storage and unique identifier assignment.
- `autogen`: Handles AI model configurations.
- `conversation_manager`, `customer_agent_creator`, `index_manager`, `rag_service_agent_creator`, `simple_service_agent_creator`, `som_service_agent_creator`: Implements different AI agents and conversation functionalities. # noqa
- `loguru`: Logs execution details.
- `scenario_data`, `settings`, `utils`: Loads scenario data and manages model settings.

Usage:
Run the script with optional command-line arguments to customize model selection, iterations, agent types, and scenario settings.

Example:
```bash
python main.py --model_name gpt-4o-mini --iterations 5 --agent_type rag --scenario default
```
"""

import argparse
import json
import os
import uuid
from datetime import datetime

import autogen
from conversation_manager import run_conversation
from customer_agent_creator import create_customer_agent, generate_initial_message
from dotenv import load_dotenv
from index_manager import get_pdf_index, get_web_index
from loguru import logger
from rag_service_agent_creator import create_rag_service_agent
from scenario_loader import (
    load_aggressive_en_scenario_data,
    load_aggressive_scenario_data,
    load_default_scenario_data,
)
from settings import configure_llm_settings
from simple_service_agent_creator import create_simple_service_agent
from som_service_agent_creator import create_society_of_mind_agent
from utils import get_client

load_dotenv()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the conversation simulation between service and customer agents."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the model to use (e.g., 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--model_provider",
        type=str,
        default="openai",
        help="Provider of the model (e.g., 'openai', 'groq').",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the simulation.",
    )
    parser.add_argument(
        "--human_input_mode_sa",
        type=str,
        default="NEVER",
        help="Human input mode for the service agent.",
    )
    parser.add_argument(
        "--human_input_mode_ca",
        type=str,
        default="NEVER",
        help="Human input mode for the customer agent.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="rag",
        help="Type of agent to use (e.g., 'rag', 'society_of_mind').",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        help="Scenario data to load (e.g., 'default', 'aggressive').",
    )
    return parser.parse_args()


def main():
    """Entry point of the program.

    This function runs a sequential conversation simulation based on the provided arguments.
    It loads a configuration list of all available models, filters the configuration based on the provided model name and provider,
    and retrieves the service and customer clients and their respective configurations.
    It also loads or creates indexes for PDF and Web tools.
    The function then iterates a specified number of times, generating scenario data for each iteration.
    For each iteration, it creates service and customer agents with system messages, generates an initial message from the customer agent,
    and runs the conversation using the service and customer agents, initial message, PDF and Web indexes, and scenario data.
    The results of each iteration are stored in a list and saved at the end.
    """
    args = parse_arguments()

    # Load configuration list of all models available to autogen
    config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    configure_llm_settings(args.model_name, "text-embedding-3-large")

    # Filter configuration based on arguments
    filter_tags = [args.model_name, args.model_provider]
    logger.info(f"Filtering configuration based on tags: {filter_tags}")
    service_client, service_config = get_client(config_list, tags=filter_tags)
    customer_client, customer_config = get_client(config_list, tags=filter_tags)
    logger.info(f"Service client: {service_client}, Customer client: {customer_client}")
    # Load or create the indexes for PDF and Web tools once for all runs
    pdf_index = get_pdf_index()
    web_index = get_web_index()

    # List to store all the results
    results = []
    logger.info(
        f"Running {args.iterations} iterations of the conversation simulation..."
    )
    for _ in range(args.iterations):
        # Sample scenario data for each run
        if args.scenario == "aggressive_en":
            scenario_data = load_aggressive_en_scenario_data()
        elif args.scenario == "aggressive":
            scenario_data = load_aggressive_scenario_data()
        else:
            scenario_data = load_default_scenario_data()
        logger.info(f"Scenario data loaded for {args.scenario} scenario.")
        # Create agents with the freshly sampled system messages and assign necessary tools
        if args.agent_type == "rag":
            service_agent, service_agent_prompt = create_rag_service_agent(
                scenario_data,
                pdf_index,
                web_index,
                args.human_input_mode_sa,
                args.scenario,
            )
        elif args.agent_type == "society_of_mind":
            service_agent, service_agent_prompt = create_society_of_mind_agent(
                scenario_data,
                pdf_index,
                web_index,
                service_config,
                args.human_input_mode_sa,
                args.scenario,
            )
        else:
            service_agent, service_agent_prompt = create_simple_service_agent(
                scenario_data,
                args.human_input_mode_sa,
                args.scenario,
            )
        logger.info(f"Service agent created with type: {args.agent_type}")
        customer_agent, customer_agent_prompt = create_customer_agent(
            scenario_data,
            args.human_input_mode_ca,
            args.scenario,
        )
        logger.info("Customer agent created.")
        # Generate the initial message from the customer agent
        initial_message = generate_initial_message(
            customer_client,
            customer_config,
            scenario_data["selected_customer_name"],
            scenario_data["selected_bank"],
            scenario_data["selected_media_type"],
            scenario_data["selected_task"],
            args.scenario,
        )
        logger.info("Initial message generated.")
        # Run the conversation
        result = run_conversation(
            service_agent,
            customer_agent,
            initial_message,
            scenario_data,
            args.agent_type,
            service_agent_prompt,
            customer_agent_prompt,
        )
        logger.info("Conversation simulation completed.")
        results.append(result)

    # Save the results
    save_results(results, args)


def save_results(results, args):
    """Save results to a JSON file, adding a unique call_id to each simulated conversation."""
    # Add a unique call_id to each conversation using a UUID
    for conversation in results:
        conversation["call_id"] = str(uuid.uuid4())

    if args.scenario == "aggressive_en":
        output_dir = "agentic_simulation_outputs/aggressive_en"
    elif args.scenario == "aggressive":
        output_dir = "agentic_simulation_outputs/aggressive"
    else:
        output_dir = "agentic_simulation_outputs/default"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"conversations-{args.model_name}-{args.agent_type}-{timestamp}.json",
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info(f"Conversations and input data have been saved to {output_file}")


if __name__ == "__main__":
    main()
