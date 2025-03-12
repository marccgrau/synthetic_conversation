import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()


def evaluate_conversation(conversation: Dict, client: OpenAI) -> float:
    """
    Use LLM to evaluate a conversation and return a quality score.
    """
    evaluation_prompt = """
    Evaluate this conversation between a call center agent and a customer and rate it on the following criteria:
    1. Realism - How well does it reflect a real-life scenario?
    2. Correctness - Does the call center agent answer all questions from the perspective as an employee of the company?
    3. Consistency - Are the agents consistent in their role and responses?
    4. Factuality - Is the information provided accurate, realistic and suitable to the personas the agents are representing?
    5. Referencing - Does the agent refer to the company's website, policies, coworkers, or other relevant information correctly as an employee of the company?

    Think step by step and reason about the conversation considering the criteria above.
    Finally, deduct an overall score (1-10) that weighs all these factors.
    Return only the score as a float with 1 decimal place.
    Make absolutely sure to only return the score, no other text.

    Conversation to evaluate:
    {conversation}
    """  # noqa: E501

    # Extract just the messages for evaluation
    messages_content = "\n".join(
        [f"{m['role']}: {m['content']}" for m in conversation["messages"]]
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt.format(conversation=messages_content),
            }
        ],
        temperature=0.0,
    )
    logger.info(f"Response: {response}")
    # Extract overall score from response
    response_text = response.choices[0].message.content
    logger.info(f"Response: {response_text}")
    try:
        # Check if the response is just the score
        response_lines = response_text.strip().split("\n")
        if len(response_lines) == 1 and response_lines[0].replace(".", "", 1).isdigit():
            overall_score = float(response_lines[0])
        else:
            # Try to extract the score from the text
            score_found = False
            for line in response_lines:
                try:
                    overall_score = float(line.strip())
                    score_found = True
                    break
                except ValueError:
                    continue
            if not score_found:
                overall_score = 0.0  # Default score if parsing fails
    except (ValueError, IndexError):
        overall_score = 0.0  # Default score if parsing fails

    return overall_score


def filter_conversations(
    input_dir: str, output_dir: str, media_type: str = None, top_n: int = 20
) -> None:
    """
    Filter conversations based on quality and media type.
    """
    # Initialize OpenAI client
    client = OpenAI()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    all_conversations = []

    # Read all json files in input directory
    json_files = glob.glob(os.path.join(input_dir, "conversations*.json"))
    logger.info(f"Found {len(json_files)} files in {input_dir}")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                conversations = json.load(f)
            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
            # Filter by media type if specified
            if media_type:
                conversations = [
                    conv
                    for conv in conversations
                    if conv["input_settings"]["selected_media_type"] == media_type
                ]

            all_conversations.extend(conversations)

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            continue
    logger.info(f"All conversations: {len(all_conversations)}")
    # Evaluate each conversation
    evaluated_conversations = []
    for conv in all_conversations:
        try:
            score = evaluate_conversation(conv, client)
            logger.info(f"Evaluated conversation {conv['call_id']} with score {score}")
            conv["llm_rating"] = score  # Add the score to the conversation
            evaluated_conversations.append((score, conv))
        except Exception as e:
            print(f"Error evaluating conversation: {str(e)}")
            continue

    # Sort by score and take top N
    evaluated_conversations.sort(reverse=True, key=lambda x: x[0])
    top_conversations = [conv for _, conv in evaluated_conversations[:top_n]]

    # Save filtered conversations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"filtered_conversations-{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top_conversations, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Filter and evaluate conversations")
    parser.add_argument(
        "--media-type", type=str, help="Filter by media type (e.g., email, phone call)"
    )
    args = parser.parse_args()

    input_dir = "agentic_simulation_outputs/default"
    output_dir = "agentic_simulation_outputs/default/filtered_conversations"

    filter_conversations(input_dir, output_dir, args.media_type)


if __name__ == "__main__":
    main()
