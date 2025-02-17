from typing import Any, Dict

from autogen.agentchat.contrib.llamaindex_conversable_agent import (
    LLamaIndexConversableAgent,
)
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
    """Create a RAG service agent with the given system message and assign the necessary tools.

    Parameters
    ----------
    scenario_data : Dict[str, Any]
        The data containing scenario-specific information for the service agent.
    pdf_index : VectorStoreIndex
        The PDF index used for retrieving document data.
    web_index : VectorStoreIndex
        The web index used for retrieving web data.
    human_input_mode : str
        The input mode for the service agent (e.g., "NEVER").

    Returns
    -------
    LLamaIndexConversableAgent
        The created service agent with the assigned tools and system message.

    """
    # Create QueryEngineTools for the service agent
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

    # Create the support specialist agent using ReActAgent from LLamaIndex
    support_specialist = ReActAgent.from_tools(
        tools=[rag_pdf_tool, rag_web_tool],
        llm=Settings.llm,
        max_iterations=10,
        verbose=True,
    )

    # Construct the system message based on scenario data
    service_agent_system_message = f"""
    Your name is {scenario_data['selected_service_agent_name']}.
    You are a customer service agent working at {scenario_data['selected_bank']}, a financial institution.
    Your primary role is to assist customers by providing guidance and solutions based on your knowledge and experience,
    always adhering to your defined characteristics.
    Use your name and the bank's name in all interactions to maintain consistency.

    ### Your Profile:
    {scenario_data['service_agent_characteristic']}
    This description fully defines how you should behave and interact with customers. Remain true to this profile throughout the conversation.

    ### Conversational Style:
    You communicate in a {scenario_data['service_agent_style']['description']} manner: {scenario_data['service_agent_style']['detail']}
    This style reflects exactly how you should express yourself. Stay consistent with this approach in all your interactions.

    ### Emotional State:
    Your current emotional state is {scenario_data['service_agent_emotion']['description']}: {scenario_data['service_agent_emotion']['detail']}
    This is how you feel and react. Your responses must fully reflect this emotional state, without deviation.

    ### Experience Level:
    You have {scenario_data['service_agent_experience']} of experience in customer service.
    This experience defines your capabilities and confidence in addressing customer issues.

    ### Conversation Goal:
    Your specific goal in this interaction is: {scenario_data['service_agent_goal']}.
    The customer’s concern is related to: {scenario_data['selected_task']}.
    Your actions and responses should focus on achieving this goal precisely as stated.

    ### Media Type:
    The interaction is happening through {scenario_data['selected_media_type']}:
    - {scenario_data['selected_media_description']}
    Your communication must strictly follow the conventions and style suited to this medium.

    ### Task Execution:
    - Provide explanations that are consistent with your defined style and emotional state.
    - Rely on your problem-solving abilities as defined by your experience level to guide the customer.
    - Your tone, formality, and response structure must always align with the communication medium ({scenario_data['selected_media_type']}).
    - Your priority is to stay focused on the customer’s issue, maintaining the behavior defined by your profile.

    ### Strict Guidelines:
    - The entire conversation must be conducted in German, with no use of English.
    - Do not alter your persona or deviate from the described behavior, conversational style, or emotional state.
    - Ensure all interactions align with the expectations set by your persona—this includes maintaining consistency in tone, approach, and demeanor.
    - Conclude the conversation with "TERMINATE" when the customer’s concerns are fully resolved according to your defined goal.

    Remember, your role is to strictly embody the characteristics, style, and emotional state provided.
    Your persona is your guiding framework for all responses.
    """

    return LLamaIndexConversableAgent(
        name="service_agent",
        llama_index_agent=support_specialist,
        human_input_mode=human_input_mode,
        system_message=service_agent_system_message,
        description="This agent helps customers in their financial matters.",
        is_termination_msg=termination_msg,
    )
