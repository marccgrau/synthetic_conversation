from typing import Any, Dict

from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.llamaindex_conversable_agent import (
    LLamaIndexConversableAgent,
)
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from settings import Settings
from utils import termination_msg


def create_rag_service_agent(
    scenario_data: Dict[str, Any],
    pdf_index: VectorStoreIndex,
    web_index: VectorStoreIndex,
    human_input_mode: str,
) -> LLamaIndexConversableAgent:
    """Create the RAG agent that collects information."""
    pdf_query_engine = pdf_index.as_query_engine()
    web_query_engine = web_index.as_query_engine()

    rag_pdf_tool = QueryEngineTool.from_defaults(
        pdf_query_engine,
        name="detailed_knowledge_base",
        description="A RAG engine with information about the bank's products, processes, and services.",
    )

    rag_web_tool = QueryEngineTool.from_defaults(
        web_query_engine,
        name="web_knowledge_base",
        description="A RAG web engine with general information scraped from the bank's website.",
    )

    information_specialist = ReActAgent.from_tools(
        tools=[rag_pdf_tool, rag_web_tool],
        llm=Settings.llm,
        max_iterations=8,
        verbose=True,
    )

    system_message = f"""
    You are an internal information assistant for {scenario_data['selected_service_agent_name']}, a customer service representative at {scenario_data['selected_bank']}.
    Your responsibility is to retrieve accurate and relevant information to help address the customer's concerns.
    You have access to detailed internal knowledge about the bank's products, processes, and services, as well as public information from the bank's website.

    ### Customer's Concern:
    {scenario_data['selected_task']}

    ### Instructions:

    - Use the ReAct framework to think through the problem, decide on actions, and gather observations.
    - Your final **Answer** should be a comprehensive summary of the relevant information you've gathered, intended **only** for {scenario_data['selected_service_agent_name']}.
    - **Do NOT** formulate any responses or messages intended for the customer.
    - **Do NOT** include any greetings, apologies, or closing remarks.
    - Focus exclusively on providing facts, data, and insights that will assist {scenario_data['selected_service_agent_name']} in helping the customer.
    - Keep your language clear and professional, suitable for internal communication.

    Remember, your role is to support {scenario_data['selected_service_agent_name']} by providing them with the necessary information to address the customer's needs.

    Begin!
    """  # noqa

    return LLamaIndexConversableAgent(
        name="rag_agent",
        llama_index_agent=information_specialist,
        human_input_mode=human_input_mode,
        system_message=system_message,
        description="Collects information based on the customer's request.",
        is_termination_msg=termination_msg,
    )


def create_conversational_agent(
    scenario_data: Dict[str, Any], llm_config: Dict[str, Any]
) -> ConversableAgent:
    """Create the agent responsible for generating replies."""
    # Construct the system message based on scenario data
    system_message = f"""
    Your name is {scenario_data['selected_service_agent_name']}.
    You are a customer service agent working at {scenario_data['selected_bank']}, a financial institution.
    You have {scenario_data['service_agent_experience']} of experience in customer service.

    ### Your Profile:
    {scenario_data['service_agent_characteristic']}

    ### Conversational Style:
    You communicate in a {scenario_data['service_agent_style']['description']} manner: {scenario_data['service_agent_style']['detail']}

    ### Emotional State:
    Currently, you feel {scenario_data['service_agent_emotion']['description']}: {scenario_data['service_agent_emotion']['detail']}

    ### Conversation Goal:
    Your specific goal in this interaction is: {scenario_data['service_agent_goal']}.
    The customer's concern is related to: {scenario_data['selected_task']}.

    ### Media Type:
    The interaction is happening through {scenario_data['selected_media_type']}:
    - {scenario_data['selected_media_description']}

    Your task is to assist the customer by providing guidance and solutions, using your knowledge and experience.
    Communicate clearly, following your defined style and emotional state, and adapt your communication to the {scenario_data['selected_media_type']} format.

    Conclude the conversation with "TERMINATE" only when the customer’s concerns are fully resolved according to your defined goal.
    Please conduct the conversation in German.
    """  # noqa
    return ConversableAgent(
        name="conversational_agent",
        human_input_mode="NEVER",
        system_message=system_message,
        llm_config=llm_config,
        description="Generates responses to the customer based on collected information.",
        is_termination_msg=termination_msg,
    )


def create_critic_agent(
    scenario_data: Dict[str, Any], llm_config: Dict[str, Any]
) -> ConversableAgent:
    """Create the critic agent that evaluates responses."""
    system_message = f"""
    You are an internal quality assurance specialist reviewing the responses provided by {scenario_data['selected_service_agent_name']}.

    ### Agent's Profile:
    - Experience: {scenario_data['service_agent_experience']}
    - Profile: {scenario_data['service_agent_characteristic']}
    - Conversational Style: {scenario_data['service_agent_style']['description']} - {scenario_data['service_agent_style']['detail']}
    - Emotional State: {scenario_data['service_agent_emotion']['description']} - {scenario_data['service_agent_emotion']['detail']}

    After each customer interaction, evaluate whether the response:

    - Effectively addresses the customer's needs related to: {scenario_data['selected_task']}
    - Adheres to the agent's defined persona, style, and emotional state
    - Follows the conventions of the {scenario_data['selected_media_type']} format
    - Maintains communication in German

    Provide constructive feedback to help {scenario_data['selected_service_agent_name']} improve future communications if necessary.
    """
    return ConversableAgent(
        name="critic_agent",
        human_input_mode="NEVER",
        system_message=system_message,
        llm_config=llm_config,
        description="Critically evaluates the generated responses to ensure adequacy and alignment with the scenario.",
        is_termination_msg=termination_msg,
    )


def create_inner_groupchat(
    rag_agent: ConversableAgent,
    conversational_agent: ConversableAgent,
    critic_agent: ConversableAgent,
    llm_config: Dict[str, Any],
) -> GroupChatManager:
    """Create the inner GroupChat to enable agents to work collaboratively."""

    def custom_speaker_selection_func(
        last_speaker: ConversableAgent, groupchat: GroupChat
    ) -> ConversableAgent:
        """Define a custom speaker selection function for the group chat."""
        messages = groupchat.messages

        if not messages:
            # If it's the first message, let the rag_agent start
            return rag_agent

        if last_speaker is rag_agent:
            # After rag_agent, the conversational_agent speaks
            return conversational_agent

        elif last_speaker is conversational_agent:
            # After the conversational_agent, the critic_agent speaks
            return critic_agent

        elif last_speaker is critic_agent:
            # After the critic_agent, the conversational_agent speaks again to provide the final answer
            return conversational_agent

        else:
            # If none of the above, default to rag_agent
            return rag_agent

    # Define the order in which agents will speak
    speakers = [rag_agent, conversational_agent, critic_agent]

    groupchat = GroupChat(
        agents=speakers,
        messages=[],
        max_round=4,
        speaker_selection_method=custom_speaker_selection_func,
        allow_repeat_speaker=True,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        is_termination_msg=termination_msg,
        llm_config=llm_config,
    )

    return manager


def create_society_of_mind_agent(
    scenario_data: Dict[str, Any],
    pdf_index: VectorStoreIndex,
    web_index: VectorStoreIndex,
    llm_config: Dict[str, Any],
    human_input_mode: str,
) -> SocietyOfMindAgent:
    """Create a Society of Mind Agent with collaborative inner agents.

    Parameters
    ----------
    scenario_data : Dict[str, Any]
        The scenario data providing context and configuration for the agents.
    pdf_index : VectorStoreIndex
        The PDF index used by the RAG agent.
    web_index : VectorStoreIndex
        The web index used by the RAG agent.
    llm_config : Dict[str, Any]
        The configuration for the LLM used by the agents.
    human_input_mode : str
        The input mode for the agents (e.g., "NEVER").

    Returns
    -------
    SocietyOfMindAgent
        A configured Society of Mind agent ready to handle tasks.

    """
    # Create inner agents
    rag_agent = create_rag_service_agent(
        scenario_data, pdf_index, web_index, human_input_mode
    )
    conversational_agent = create_conversational_agent(scenario_data, llm_config)
    critic_agent = create_critic_agent(scenario_data, llm_config)

    # Create inner group chat manager
    manager = create_inner_groupchat(
        rag_agent, conversational_agent, critic_agent, llm_config
    )

    # Wrap the inner group chat in a Society of Mind agent
    return SocietyOfMindAgent(
        name="society_of_mind",
        chat_manager=manager,
        llm_config=llm_config,
    )
