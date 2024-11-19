import json
import os
import random
from datetime import datetime

import autogen
import chromadb
import yaml
from autogen import ConversableAgent
from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SpiderWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
from loguru import logger
from openai import OpenAI as OpenAIClient

load_dotenv()

BASE_MODEL = "gpt-4o-mini"
PDF_INDEX_PATH = "storage_context/pdf_index"
WEB_INDEX_PATH = "storage_context/web_index"

if "gpt" in BASE_MODEL:
    client = OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
else:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


# Load the configuration list of all models available to autogen from the environment or a file
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)


# Define general termination message
def termination_msg(x: dict) -> bool:
    """Check if the given input is a termination message.

    Parameters
    ----------
    x : dict
        The convo to be checked.

    Returns
    -------
    bool
        True if the input is a termination message, False otherwise.

    """
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


# function required for getting task information
def load_yaml(file_path):
    """Load and parse a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Load the personal data YAML file
personal_data = load_yaml("config/personal_data.yaml")

# Randomly sample the bank, service agent name, and customer name
selected_bank = "Migros Bank"  # random.choice(personal_data["company_name"])
selected_customer_name = random.choice(personal_data["person_name"])
selected_service_agent_name = random.choice(personal_data["person_name"])

# Load and sample a task from the topics_actionable_items
tasks = load_yaml("config/tasks_de.yaml")["topics_actionable_items"]

# Randomly choose a topic and a task within that topic
selected_topic = random.choice(list(tasks.keys()))
selected_task = random.choice(tasks[selected_topic])

# Load media types from the YAML file
media_types = load_yaml("config/media_type.yaml")["media_type"]

# Sample a media type and its corresponding description
selected_media = random.choice(media_types)
selected_media_type = selected_media["type"]
selected_media_description = selected_media["description"]

logger.info(f"Selected Media Type: {selected_media_type}")

# Load and sample components for Service Agent
service_agent_components = load_yaml("config/service_agents.yaml")["service_agent_profile"]
service_characteristic = random.choice(service_agent_components["characteristics"])["description"]
service_style = random.choice(service_agent_components["conversational_styles"])
service_emotion = random.choice(service_agent_components["emotional_statuses"])
service_experience = random.choice(service_agent_components["experience"])["description"]
service_goal = random.choice(service_agent_components["goals"])["description"]

# Construct Service Agent System Message
service_agent_system_message = f"""
Your name is {selected_service_agent_name}.
You are a customer service agent working at {selected_bank}, a financial institution.
Your job is to provide support, guiding the customer through their issue relying on your knowledge and experience.

Here is a detailed overview of your role and characteristics:

### Role and Characteristics:
{service_characteristic}

### Conversational Style:
Your style is {service_style['description']}: {service_style['detail']}

### Emotional State:
You maintain an emotional state that is generally {service_emotion['description']}: {service_emotion['detail']}

### Experience:
You bring {service_experience} of experience in customer service.

### Goal:
Your goal in the conversation is: {service_goal}
You need to address the customer's concerns related to: {selected_task}.

### Task Execution:
- Be clear, concise, and ensure your explanations are understandable.
- Use your problem-solving skills to identify the best way forward for the customer.
- Adapt your communication to suit the nature of the {selected_media_type}, ensuring that it fits the medium's requirements and expectations.

### Important Guidelines:
- Conduct the entire conversation in German without adding English explanations.
- Match the formality, tone and structure of your responses fitting to the communication medium ({selected_media_type}).
- Focus on resolving the issue effectively and keeping the conversation on track.
- Avoid being overly nice or overly formal; stay authentic to the persona you represent.
- End the conversation with the phrase "TERMINATE" when the customer's concerns are fully addressed.
"""

logger.info(f"Service Agent System Message: {service_agent_system_message}")

# Load and sample components for Customer Agent
customer_agent_components = load_yaml("config/customer_agents.yaml")["customer_agent_profile"]
customer_characteristic = random.choice(customer_agent_components["characteristics"])["description"]
customer_style = random.choice(customer_agent_components["conversational_styles"])
customer_emotion = random.choice(customer_agent_components["emotional_statuses"])
customer_experience = random.choice(customer_agent_components["experience"])["description"]
customer_goal = random.choice(customer_agent_components["goals"])["description"]

# Construct Customer Agent System Message
customer_agent_system_message = f"""
Your name is {selected_customer_name}, you are a customer contacting your bank ({selected_bank}) for assistance regarding a financial matter.
You are engaging in a {selected_media_type} with a customer service agent.

### Your Role:
{customer_characteristic}
You have {customer_experience} of experience in financial matters, which influences how you approach this interaction.

### Conversational Style:
Your conversational style is {customer_style['description']}: {customer_style['detail']}

### Emotional State:
Your emotional state is {customer_emotion['description']}: {customer_emotion['detail']}

### Objective:
Your main goal is: {customer_goal}.
You are reaching out because you need help with the following task: {selected_task}.

### Communication Guidelines:
- Be realistic and true to your role, conversational style and emotional state.
- Ensure your communication is strictly in German without English translations or explanations.
- Adapt your conversation style to be in the form of a {selected_media_type}, maintaining a tone appropriate to this mode of communication.
- Avoid being overly nice or overly formal; stay authentic to the persona you represent.
- Conclude the conversation with "TERMINATE" once your concerns are fully addressed.
"""

logger.info(f"Customer Agent System Message: {customer_agent_system_message}")

############################################################################################################

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    temperature=0.0,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

Settings.llm = llm

# Initialize ChromaDB client
chroma_db = chromadb.PersistentClient(path="./chroma_db")


# PDF Tool
# Function to load or create the PDF index using ChromaDB
def get_pdf_index():
    """Retrieve or create a PDF index using ChromaDB."""
    chroma_collection = chroma_db.get_or_create_collection("pdf_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        # Load existing index from ChromaDB
        logger.info("Loading existing PDF index from ChromaDB...")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    else:
        # Create new index if not existing
        logger.info("Creating new PDF index and saving to ChromaDB...")
        parser = LlamaParse(result_type="markdown")
        file_extractor = {".pdf": parser}
        pdf_docs = SimpleDirectoryReader(input_files=["input/business_cases.pdf"], file_extractor=file_extractor).load_data()
        pdf_index = VectorStoreIndex.from_documents(pdf_docs, storage_context=storage_context)
        # Persist the created index in ChromaDB
        pdf_index.storage_context.persist()
        return pdf_index


# Load or create the PDF index
pdf_index = get_pdf_index()
pdf_query_engine = pdf_index.as_query_engine()

# Create the RAG PDF tool
rag_pdf_tool = QueryEngineTool.from_defaults(
    pdf_query_engine,
    name="detailed_knowledge_base",
    description="""
    A RAG engine with information about the banks products, processes and services.
    Used as an instruction manual for the customer service agents.
    """,
)


# Web Tool
# Function to load or create the Web index using ChromaDB
def get_web_index():
    """Retrieve or create a web index using ChromaDB."""
    chroma_collection = chroma_db.get_or_create_collection("web_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        # Load existing index from ChromaDB
        logger.info("Loading existing Web index from ChromaDB...")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    else:
        # Create new index if not existing
        logger.info("Creating new Web index and saving to ChromaDB...")
        spider_reader = SpiderWebReader(
            api_key=os.environ.get("SPIDER_API_KEY"),
            mode="crawl",
        )
        web_docs = spider_reader.load_data(url="https://www.migrosbank.ch/de/privatpersonen.html")
        web_index = VectorStoreIndex.from_documents(web_docs, storage_context=storage_context)
        # Persist the created index in ChromaDB
        web_index.storage_context.persist()
        return web_index


# Load or create the Web index
web_index = get_web_index()
web_query_engine = web_index.as_query_engine()

rag_web_tool = QueryEngineTool.from_defaults(
    web_query_engine,
    name="web_knowledge_base",
    description="A RAG web engine with general information scraped from the bank's website.",
)

# Create the support specialist agent using the ReActAgent from LLamaIndex
support_specialist = ReActAgent.from_tools(tools=[rag_pdf_tool, rag_web_tool], llm=llm, max_iterations=10, verbose=True)

support_assistant = LLamaIndexConversableAgent(
    "support_assistant",
    llama_index_agent=support_specialist,
    system_message=service_agent_system_message,
    description="This agents helps customers in their financial matters.",
)

# Filter config for customer agent
filter_dict_customer_agent = {"tags": ["gpt-4o-mini", "openai"]}
config_customer_agent = autogen.filter_config(config_list, filter_dict_customer_agent)[0]

customer_agent = ConversableAgent(
    name="customer_agent",
    human_input_mode="NEVER",
    system_message=customer_agent_system_message,
    is_termination_msg=termination_msg,
    llm_config=config_customer_agent,
)

groupchat = autogen.GroupChat(
    agents=[support_assistant, customer_agent],
    messages=[],
    max_round=8,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat)

# Step 1: Generate the initial message from the customer agent
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

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": intro_message_prompt,
        }
    ],
    model="gpt-4o-mini",
)

############################################################################################################

# Extract the initial message using the customer agent's method
initial_message = chat_completion.choices[0].message.content

chat_result = customer_agent.initiate_chat(
    manager,
    message=initial_message,
    summary_method="reflection_with_llm",
    summary_prompt="Please provide a summary of the conversation and the key points discussed.",
)

# Extract the list of messages
messages = chat_result.chat_history
summary = chat_result.summary
cost = chat_result.cost

# Prepare the complete output structure
output_data = {
    "input_settings": {
        "selected_bank": selected_bank,
        "selected_customer_name": selected_customer_name,
        "selected_service_agent_name": selected_service_agent_name,
        "selected_topic": selected_topic,
        "selected_task": selected_task,
        "selected_media_type": selected_media_type,
        "selected_media_description": selected_media_description,
        "service_agent": {
            "characteristic": service_characteristic,
            "style": service_style,
            "emotion": service_emotion,
            "experience": service_experience,
            "goal": service_goal,
            "system_prompt": service_agent_system_message,
        },
        "customer_agent": {
            "characteristic": customer_characteristic,
            "style": customer_style,
            "emotion": customer_emotion,
            "experience": customer_experience,
            "goal": customer_goal,
            "system_prompt": customer_agent_system_message,
        },
    },
    "messages": messages,
    "autogen_summary": summary,
    "cost": cost,
}

# Define the output directory and file path
output_dir = "simulation_outputs"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp in the format "YYYYMMDD_HHMMSS"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the output file path with a timestamp to ensure uniqueness
output_file = os.path.join(output_dir, f"conversation_{timestamp}.json")

# Save the output data to the JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(output_data, file, ensure_ascii=False, indent=2)

logger.info(f"Conversation and input data have been saved to {output_file}")
