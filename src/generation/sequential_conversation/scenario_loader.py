import random
from typing import Any, Dict

from utils import load_yaml


def load_default_scenario_data() -> Dict[str, Any]:
    """Sample and load scenario data for the conversation simulation.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing sampled scenario data including agent characteristics, task details,
        and system messages for both service and customer agents.

    """
    # Load personal data
    personal_data: Dict[str, Any] = load_yaml("config/default/personal_data.yaml")

    # Randomly sample the bank, service agent name, and customer name
    selected_bank: str = random.choice(personal_data["company_name"])
    selected_customer_name: str = random.choice(personal_data["person_name"])
    selected_service_agent_name: str = random.choice(personal_data["person_name"])

    # Load and sample a task from the topics_actionable_items
    tasks: Dict[str, Any] = load_yaml("config/tasks_de.yaml")["topics_actionable_items"]
    selected_topic: str = random.choice(list(tasks.keys()))
    selected_task: str = random.choice(tasks[selected_topic])

    # Load media types and sample
    media_types: list[Dict[str, Any]] = load_yaml("config/media_type.yaml")[
        "media_type"
    ]
    selected_media: Dict[str, str] = random.choice(media_types)
    selected_media_type: str = selected_media["type"]
    selected_media_description: str = selected_media["description"]

    # Load and sample components for Service Agent
    service_agent_components: Dict[str, Any] = load_yaml(
        "config/default/service_agents.yaml"
    )["service_agent_profile"]
    service_characteristic: str = random.choice(
        service_agent_components["characteristics"]
    )["description"]
    service_style: Dict[str, str] = random.choice(
        service_agent_components["conversational_styles"]
    )
    service_emotion: Dict[str, str] = random.choice(
        service_agent_components["emotional_statuses"]
    )
    service_experience: str = random.choice(service_agent_components["experience"])[
        "description"
    ]
    service_goal: str = random.choice(service_agent_components["goals"])["description"]

    # Load and sample components for Customer Agent
    customer_agent_components: Dict[str, Any] = load_yaml(
        "config/default/customer_agents.yaml"
    )["customer_agent_profile"]
    customer_characteristic: str = random.choice(
        customer_agent_components["characteristics"]
    )["description"]
    customer_style: Dict[str, str] = random.choice(
        customer_agent_components["conversational_styles"]
    )
    customer_emotion: Dict[str, str] = random.choice(
        customer_agent_components["emotional_statuses"]
    )
    customer_experience: str = random.choice(customer_agent_components["experience"])[
        "description"
    ]
    customer_goal: str = random.choice(customer_agent_components["goals"])[
        "description"
    ]

    return {
        "selected_bank": selected_bank,
        "selected_customer_name": selected_customer_name,
        "selected_service_agent_name": selected_service_agent_name,
        "selected_task": selected_task,
        "selected_media_type": selected_media_type,
        "selected_media_description": selected_media_description,
        "service_agent_characteristic": service_characteristic,
        "service_agent_style": service_style,
        "service_agent_emotion": service_emotion,
        "service_agent_experience": service_experience,
        "service_agent_goal": service_goal,
        "customer_agent_characteristic": customer_characteristic,
        "customer_agent_style": customer_style,
        "customer_agent_emotion": customer_emotion,
        "customer_agent_experience": customer_experience,
        "customer_agent_goal": customer_goal,
    }


def load_aggressive_scenario_data() -> Dict[str, Any]:
    """Sample and load scenario data for the conversation simulation.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing sampled scenario data including agent characteristics, task details,
        and system messages for both service and customer agents.

    """
    # Load personal data
    personal_data: Dict[str, Any] = load_yaml("config/aggressive/personal_data.yaml")

    # Randomly sample the bank, service agent name, and customer name
    selected_bank: str = random.choice(personal_data["company_name"])
    selected_customer_name: str = random.choice(personal_data["person_name"])
    selected_service_agent_name: str = random.choice(personal_data["bot_name"])

    # Load and sample a task from the topics_actionable_items
    tasks: Dict[str, Any] = load_yaml("config/tasks_de.yaml")["topics_actionable_items"]
    selected_topic: str = random.choice(list(tasks.keys()))
    selected_task: str = random.choice(tasks[selected_topic])

    # Load media types and sample
    media_types: list[Dict[str, Any]] = load_yaml("config/media_type.yaml")[
        "media_type"
    ]
    selected_media: Dict[str, str] = random.choice(media_types)
    selected_media_type: str = selected_media["type"]
    selected_media_description: str = selected_media["description"]

    # Load and sample components for Service Agent
    service_agent_components: Dict[str, Any] = load_yaml(
        "config/aggressive/service_agents.yaml"
    )["service_agent_profile"]
    service_characteristic: str = random.choice(
        service_agent_components["characteristics"]
    )["description"]
    service_style: Dict[str, str] = random.choice(
        service_agent_components["conversational_styles"]
    )
    service_emotion: Dict[str, str] = random.choice(
        service_agent_components["emotional_statuses"]
    )
    service_experience: str = random.choice(service_agent_components["experience"])[
        "description"
    ]
    service_goal: str = random.choice(service_agent_components["goals"])["description"]

    # Load and sample components for Customer Agent
    customer_agent_components: Dict[str, Any] = load_yaml(
        "config/aggressive/customer_agents.yaml"
    )["customer_agent_profile"]
    customer_characteristic: str = random.choice(
        customer_agent_components["characteristics"]
    )["description"]
    customer_style: Dict[str, str] = random.choice(
        customer_agent_components["conversational_styles"]
    )
    customer_emotion: Dict[str, str] = random.choice(
        customer_agent_components["emotional_statuses"]
    )
    customer_experience: str = random.choice(customer_agent_components["experience"])[
        "description"
    ]
    customer_goal: str = random.choice(customer_agent_components["goals"])[
        "description"
    ]

    return {
        "selected_bank": selected_bank,
        "selected_customer_name": selected_customer_name,
        "selected_service_agent_name": selected_service_agent_name,
        "selected_task": selected_task,
        "selected_media_type": selected_media_type,
        "selected_media_description": selected_media_description,
        "service_agent_characteristic": service_characteristic,
        "service_agent_style": service_style,
        "service_agent_emotion": service_emotion,
        "service_agent_experience": service_experience,
        "service_agent_goal": service_goal,
        "customer_agent_characteristic": customer_characteristic,
        "customer_agent_style": customer_style,
        "customer_agent_emotion": customer_emotion,
        "customer_agent_experience": customer_experience,
        "customer_agent_goal": customer_goal,
    }


def load_aggressive_en_scenario_data() -> Dict[str, Any]:
    """Sample and load scenario data for the conversation simulation.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing sampled scenario data including agent characteristics, task details,
        and system messages for both service and customer agents.

    """
    # Load personal data
    personal_data: Dict[str, Any] = load_yaml("config/aggressive/personal_data.yaml")

    # Randomly sample the bank, service agent name, and customer name
    selected_bank: str = random.choice(personal_data["company_name"])
    selected_customer_name: str = random.choice(personal_data["person_name"])
    selected_service_agent_name: str = random.choice(personal_data["bot_name"])

    # Load and sample a task from the topics_actionable_items
    tasks: Dict[str, Any] = load_yaml("config/tasks_en.yaml")["topics_actionable_items"]
    selected_topic: str = random.choice(list(tasks.keys()))
    selected_task: str = random.choice(tasks[selected_topic])

    # Load media types and sample
    media_types: list[Dict[str, Any]] = load_yaml("config/media_type.yaml")[
        "media_type"
    ]
    selected_media: Dict[str, str] = next(
        mt for mt in media_types if mt["type"] == "phone call"
    )
    selected_media_type: str = selected_media["type"]
    selected_media_description: str = selected_media["description"]

    # Load and sample components for Service Agent
    service_agent_components: Dict[str, Any] = load_yaml(
        "config/aggressive/service_agents.yaml"
    )["service_agent_profile"]
    service_characteristic: str = random.choice(
        service_agent_components["characteristics"]
    )["description"]
    service_style: Dict[str, str] = random.choice(
        service_agent_components["conversational_styles"]
    )
    service_emotion: Dict[str, str] = random.choice(
        service_agent_components["emotional_statuses"]
    )
    service_experience: str = random.choice(service_agent_components["experience"])[
        "description"
    ]
    service_goal: str = random.choice(service_agent_components["goals"])["description"]

    # Load and sample components for Customer Agent
    customer_agent_components: Dict[str, Any] = load_yaml(
        "config/aggressive/customer_agents.yaml"
    )["customer_agent_profile"]
    customer_characteristic: str = random.choice(
        customer_agent_components["characteristics"]
    )["description"]
    customer_style: Dict[str, str] = random.choice(
        customer_agent_components["conversational_styles"]
    )
    customer_emotion: Dict[str, str] = random.choice(
        customer_agent_components["emotional_statuses"]
    )
    customer_experience: str = random.choice(customer_agent_components["experience"])[
        "description"
    ]
    customer_goal: str = random.choice(customer_agent_components["goals"])[
        "description"
    ]

    return {
        "selected_bank": selected_bank,
        "selected_customer_name": selected_customer_name,
        "selected_service_agent_name": selected_service_agent_name,
        "selected_task": selected_task,
        "selected_media_type": selected_media_type,
        "selected_media_description": selected_media_description,
        "service_agent_characteristic": service_characteristic,
        "service_agent_style": service_style,
        "service_agent_emotion": service_emotion,
        "service_agent_experience": service_experience,
        "service_agent_goal": service_goal,
        "customer_agent_characteristic": customer_characteristic,
        "customer_agent_style": customer_style,
        "customer_agent_emotion": customer_emotion,
        "customer_agent_experience": customer_experience,
        "customer_agent_goal": customer_goal,
    }
