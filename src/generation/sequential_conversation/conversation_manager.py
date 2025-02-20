from typing import Any, Dict

from autogen import GroupChat, GroupChatManager
from autogen.agentchat import ConversableAgent
from loguru import logger
from utils import termination_msg


def run_conversation(
    service_agent: ConversableAgent,
    customer_agent: ConversableAgent,
    initial_message: str,
    data: Dict[str, Any],
    agent_type: str,
) -> Dict[str, Any]:
    """Run the simulated conversation between the service agent and customer agent.

    Parameters
    ----------
    service_agent : ConversableAgent
        The service agent handling the customer interaction.
    customer_agent : ConversableAgent
        The customer agent initiating the conversation.
    initial_message : str
        The initial message from the customer agent to start the conversation.
    data : Dict[str, Any]
        The scenario data used to construct the agents and their system messages.
    agent_type : str
        The type of agent used for the conversation (e.g., "rag", "society_of_mind", "simple").

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the conversation details, input settings, and summary of the interaction.

    """
    groupchat = GroupChat(
        agents=[service_agent, customer_agent],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",
        enable_clear_history=True,
    )
    logger.info("GroupChat initialized successfully.")
    manager = GroupChatManager(
        groupchat=groupchat,
        is_termination_msg=termination_msg,
    )
    logger.info("GroupChatManager initialized successfully.")
    summary_prompt = """
    Please provide a comprehensive summary of the conversation, including the following:

    1. **Main Objectives**: Clearly state the primary goals or issues the customer presented at the beginning.
    2. **Key Points Discussed**: Highlight the major topics or questions addressed during the conversation.
    3. **Actions Taken**: Describe any steps, guidance, or solutions provided by the service agent.
    4. **Resolution Status**: Indicate whether the customer's issue was resolved, partially resolved, or remains unresolved.
    5. **Next Steps**: List any discussed follow-up actions or next steps if applicable.

    Make sure you only include relevant information and avoid unnecessary details.

    Ensure the summary is concise yet comprehensive, capturing the essence of the interaction in a clear and structured manner.
    Write the summary in German. Return only the German text for the summary.
    """
    logger.info(f"Start Convo with initial message: {initial_message}")
    chat_result = customer_agent.initiate_chat(
        manager,
        message=initial_message,
        summary_method="reflection_with_llm",
        summary_prompt=summary_prompt,
    )

    logger.info("Conversation simulation completed successfully.")

    # Extract the list of messages
    messages = chat_result.chat_history
    summary = chat_result.summary
    cost = chat_result.cost

    # Prepare the complete output structure with all characteristics included
    return {
        "input_settings": {
            "selected_bank": data["selected_bank"],
            "selected_customer_name": data["selected_customer_name"],
            "selected_service_agent_name": data["selected_service_agent_name"],
            "selected_task": data["selected_task"],
            "selected_media_type": data["selected_media_type"],
            "selected_media_description": data["selected_media_description"],
            "service_agent": {
                "characteristic": data["service_agent_characteristic"],
                "style": data["service_agent_style"],
                "emotion": data["service_agent_emotion"],
                "experience": data["service_agent_experience"],
                "goal": data["service_agent_goal"],
            },
            "customer_agent": {
                "characteristic": data["customer_agent_characteristic"],
                "style": data["customer_agent_style"],
                "emotion": data["customer_agent_emotion"],
                "experience": data["customer_agent_experience"],
                "goal": data["customer_agent_goal"],
            },
        },
        "messages": messages,
        "summary_prompt": summary_prompt,
        "autogen_summary": summary,
        "cost": cost,
        "agent_type": agent_type,
    }
