import argparse
import json
import os
from datetime import datetime

import autogen
from conversation_manager import run_conversation
from customer_agent_creator import create_customer_agent, generate_initial_message
from index_manager import get_pdf_index, get_web_index
from loguru import logger
from rag_service_agent_creator import create_rag_service_agent
from scenario_loader import load_scenario_data  # Import the new scenario loader
from settings import configure_llm_settings
from simple_service_agent_creator import create_simple_service_agent
from som_service_agent_creator import create_society_of_mind_agent
from utils import get_client


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
    service_client, service_config = get_client(config_list, tags=filter_tags)
    customer_client, customer_config = get_client(config_list, tags=filter_tags)

    # Load or create the indexes for PDF and Web tools once for all runs
    pdf_index = get_pdf_index()
    web_index = get_web_index()

    # List to store all the results
    results = []

    for _ in range(args.iterations):
        # Sample scenario data for each run
        scenario_data = load_scenario_data()

        # Create agents with the freshly sampled system messages and assign necessary tools
        if args.agent_type == "rag":
            service_agent = create_rag_service_agent(
                scenario_data,
                pdf_index,
                web_index,
                args.human_input_mode_sa,
            )
        elif args.agent_type == "society_of_mind":
            service_agent = create_society_of_mind_agent(
                scenario_data,
                pdf_index,
                web_index,
                service_config,
                args.human_input_mode_sa,
            )
        else:
            service_agent = create_simple_service_agent(
                scenario_data,
                args.human_input_mode_sa,
            )

        customer_agent = create_customer_agent(
            scenario_data,
            args.human_input_mode_ca,
        )

        # Generate the initial message from the customer agent
        initial_message = generate_initial_message(
            customer_client,
            customer_config,
            scenario_data["selected_customer_name"],
            scenario_data["selected_bank"],
            scenario_data["selected_media_type"],
            scenario_data["selected_task"],
        )

        # Run the conversation
        result = run_conversation(
            service_agent,
            customer_agent,
            initial_message,
            scenario_data,
            args.agent_type,
        )
        results.append(result)

    # Save the results
    save_results(results)


def save_results(results):
    """Save results to a JSON file."""
    output_dir = "agentic_simulation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"conversations_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info(f"Conversations and input data have been saved to {output_file}")


if __name__ == "__main__":
    main()
