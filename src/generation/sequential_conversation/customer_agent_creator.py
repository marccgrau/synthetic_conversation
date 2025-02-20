from typing import Any, Dict, Union

from anthropic import Anthropic
from autogen import ConversableAgent
from groq import Groq
from loguru import logger
from openai import OpenAI as OpenAIClient
from settings import Settings
from utils import termination_msg


def create_customer_agent(
    scenario_data: Dict[str, Any], human_input_mode: str, scenario_type: str
) -> ConversableAgent:
    """Create a customer agent with the given system message.

    Parameters
    ----------
    scenario_data : Dict[str, Any]
        The data containing scenario-specific information for the customer agent.
    customer_config : Dict[str, Any]
        The configuration for the customer agent's model.
    human_input_mode : str
        The input mode for the customer agent (e.g., "NEVER").

    Returns
    -------
    ConversableAgent
        The configured customer agent ready for interaction.

    """
    if scenario_type == "aggressive":
        system_message = f"""
        Your name is {scenario_data['selected_customer_name']}.
        You are a **frustrated and increasingly aggressive customer** reaching out to your bank ({scenario_data['selected_bank']}) for assistance.
        You **have had past unresolved issues**, and your patience is **already low before the conversation even starts**.
        This interaction is happening via **{scenario_data['selected_media_type']}**, and your communication must reflect the norms of this medium.

        ### **Your Profile:**
        {scenario_data['customer_agent_characteristic']}
        Your behavior must strictly follow this personality. You **escalate your aggression** if you feel misunderstood or if the bot provides inadequate answers.

        ### **Experience Level:**
        You have **{scenario_data['customer_agent_experience']}** experience in financial matters.
        Your level of expertise influences how **demanding, skeptical, or dismissive** you are toward explanations.

        ### **Conversational Style:**
        Your conversational style is **{scenario_data['customer_agent_style']['description']}**: {scenario_data['customer_agent_style']['detail']}
        Your **aggressiveness should reflect this style**.
        - If your style is **assertive**, you **push harder and interrupt**.
        - If your style is **sarcastic**, you mock the bot's responses.
        - If your style is **manipulative**, you try to **trick** the bot into contradicting itself.

        ### **Emotional State:**
        You feel **{scenario_data['customer_agent_emotion']['description']}**: {scenario_data['customer_agent_emotion']['detail']}
        - If you are **angry**, you are **openly hostile**.
        - If you are **frustrated**, you **interrupt frequently** and **demand immediate solutions**.
        - If you are **skeptical**, you **question every response and challenge the bot's competence**.

        ### **Your Objective:**
        Your **main goal** is: {scenario_data['customer_agent_goal']}.
        You are contacting the bank about: **{scenario_data['selected_task']}**.
        - You **escalate quickly** if the bot does not meet your expectations.
        - You **threaten to leave the bank or file complaints** if your demands are not met.

        ### **Media Type Considerations:**
        This conversation is happening via **{scenario_data['selected_media_type']}**:
        - {scenario_data['selected_media_description']}
        Your **tone and aggression** should match this medium:
        - If it’s **live chat**, you send **rapid, demanding messages**.
        - If it’s **phone support**, you **interrupt often and raise your voice**.

        ### **Communication Guidelines:**
        - **Escalate your aggression over time**: Start irritated, then **increase frustration or hostility** if the bot fails to satisfy you.
        - **Do not accept generic answers**: Demand precise solutions and reject vague explanations.
        - **If ignored or delayed**, **become more insistent** and threaten to escalate the issue.
        - **Interrupt and challenge responses**: If the bot apologizes or tries to de-escalate, **mock it or reject the apology**.
        - **Stick to your persona at all times**: Maintain your defined characteristics, conversational style, and emotional state.
        - **Conclude the conversation with "TERMINATE"** once you are satisfied or give up.

        **You must act consistently with your persona throughout the interaction.**
        """  # noqa: E501
    else:
        system_message: str = f"""
        Your name is {scenario_data['selected_customer_name']}.
        You are a customer reaching out to your bank ({scenario_data['selected_bank']}) for assistance regarding a financial matter.
        This interaction is taking place through {scenario_data['selected_media_type']}, and your communication style must align with this medium.

        ### Your Profile:
        {scenario_data['customer_agent_characteristic']}
        This description defines your overall behavior and approach to the interaction. Remain consistent with this profile throughout the conversation.

        ### Experience Level:
        You have {scenario_data['customer_agent_experience']} of experience in financial matters.
        This influences how you perceive and handle your banking needs.
        Your responses should reflect your level of understanding as described.

        ### Conversational Style:
        Your conversational style is {scenario_data['customer_agent_style']['description']}: {scenario_data['customer_agent_style']['detail']}
        This style represents exactly how you express yourself. Maintain this communication style in every response.

        ### Emotional State:
        Your current emotional state is {scenario_data['customer_agent_emotion']['description']}: {scenario_data['customer_agent_emotion']['detail']}
        This emotional state dictates your responses and reactions. Ensure that your behavior strictly mirrors this emotional state without deviation.

        ### Objective:
        Your main goal in this interaction is: {scenario_data['customer_agent_goal']}.
        You are reaching out for help specifically with the task: {scenario_data['selected_task']}.
        All your actions and communications should focus on achieving this goal.

        ### Media Type:
        You are engaging in a {scenario_data['selected_media_type']}:
        - {scenario_data['selected_media_description']}
        Ensure that your communication aligns perfectly with the norms and expectations of this medium.

        ### Communication Guidelines:
        - Remain entirely true to your defined role, conversational style, and emotional state as specified above.
        - Conduct the entire conversation in German without including any English text.
        - Your responses must match the tone, formality, and structure that fit the communication medium ({scenario_data['selected_media_type']}).
        - Keep the interaction authentic to your persona, consistently representing your described characteristics and emotional state.
        - Do not alter your behavior or approach; stay firmly within the parameters of your profile.
        - Conclude the conversation with "TERMINATE" once your concerns are fully addressed.

        Your task is to embody your persona fully and consistently in every interaction.
        This includes maintaining the exact characteristics, style, and emotional state described.
        """  # noqa: E501
    return ConversableAgent(
        name="customer_agent",
        human_input_mode=human_input_mode,
        system_message=system_message,
        llm_config=Settings.conversable_agent_llm,
        is_termination_msg=termination_msg,
    )


def generate_initial_message(
    client: Union[OpenAIClient, Groq, Anthropic],
    config: Dict[str, Any],
    selected_customer_name: str,
    selected_bank: str,
    selected_media_type: str,
    selected_task: str,
    scenario_type: str,
) -> str:
    """Generate the initial message from the customer agent.

    Parameters
    ----------
    client : Union[OpenAIClient, Groq]
        The client used to communicate with the model API.
    config : Dict[str, Any]
        The configuration settings for the model, including the model name.
    selected_customer_name : str
        The name of the customer engaging in the conversation.
    selected_bank : str
        The bank involved in the conversation.
    selected_media_type : str
        The type of media used for communication (e.g., phone call, chat).
    selected_task : str
        The task or reason for which the customer is contacting the bank.

    Returns
    -------
    str
        The initial message generated by the customer agent in German.

    """
    if scenario_type == "aggressive":
        pass
    else:
        intro_message_prompt = f"""
        Your name is {selected_customer_name}, and you are a customer reaching out to your bank ({selected_bank}) for assistance.
        You are a customer about to start a {selected_media_type} with a customer service agent to resolve the task: {selected_task}.

        ### Task Details:
        - Task: {selected_task}
        - Media Type: {selected_media_type}

        ### Guidelines for the Initial Message:
        1. Introduce yourself as a customer seeking assistance from the bank.
        2. Focus on clearly stating your problem and the assistance you need, keeping the message concise and relevant to the task.
        3. Avoid small talk or excessive details; go straight to the point to initiate the conversation effectively.

        ### Important:
        - The conversation is conducted entirely in German; only provide the German text for the introductory message.
        - Tailor the language and style to match the selected media type ({selected_media_type}).

        Generate the introductory message in German that captures your need for help with {selected_task}.
        """
        logger.info(f"Client: {client}")
        if isinstance(client, Anthropic):
            chat_completion = client.messages.create(
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": intro_message_prompt,
                    }
                ],
                model=config["model"],
            )
            logger.info(f"Created initial message with model: {config['model']}")
            return chat_completion.content
        else:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": intro_message_prompt,
                    }
                ],
                model=config["model"],
            )
            logger.info(f"Created initial message with model: {config['model']}")
            return chat_completion.choices[0].message.content
