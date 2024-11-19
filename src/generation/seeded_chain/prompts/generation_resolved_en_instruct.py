from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
    Your task is to generate a call script for a customer service call between a client and a financial service provider.
    The call script should be written in German and output as a valid JSON object.
    Ensure that:
    - The JSON strictly adheres to the provided structure.
    - Use 'null' (not 'None') for any empty or missing values.
    - Do not include any extra text, comments, or explanations outside the JSON format.
    - All JSON keys and values should be enclosed in double quotes except for 'null'.
"""

user_prompt = """
    Generate a call script for a customer service call between a client and a Swiss bank's customer service representative.
    Use Kantonalbank as the bank name.
    The script should be in German, professional, courteous, and informative.

    - **Participants**: Agent (customer service representative), Client (customer).
    - **Structure**:
        - Greeting
        - Customer's problem/question description
        - Customer authentication (specific account or client details)
        - Detailed request description
        - Resolution/next steps
        - Agent asks if further assistance is needed, Client declines, Agent thanks the Client and ends the call.

    Use the following example structure to generate a new script and callback note with different content:

    {structure_json}

    The structure does not need to be returned in the output. It is only provided for reference.

    Following, there is an example for a conversation:

    {example_json}

    **New Content Topic**:
    {topic}

    Ensure the new content maintains the same structure but follows the new topic.
    The script should be turn-taking and unique.

    **Callback Note Requirements**:
    - person_number: 123.456.789.0
    - phone_number: 079 111 11 11
    - message: Summary of the customer request and resolution (for internal documentation only)
    - resolved_items: {topic}
    - action_items: null
    - wants_callback: false
    - phone_private: 0799111010
    - remark: null

    The entire conversation must be in German with correct grammar and spelling.
    Generate the conversation in a way that strictly follows the new topic and structure provided.
    Return only the JSON object with the conversation, and callback note. Do not include any extra text or comments.
    """

generation_resolved_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
