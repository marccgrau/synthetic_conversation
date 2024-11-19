from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
    Your task is to validate and correct the following call script.
    Ensure that the output is a valid JSON object, strictly adhering to the expected structure provided.
    The JSON object must:
    - Use double quotes for all keys and string values.
    - Represent any empty or missing values as 'null' (without quotes).
    - Contain no extra text, comments, or explanations outside the JSON format.

    Additionally, ensure that the conversation is:
    - Professional, courteous, informative, and relevant to banking services.
    - In German, with correct grammar and spelling.

    Return only the corrected JSON object, ensuring it follows all the above guidelines.
"""


user_prompt = """
    Validate and correct the following generated call script.

    {script_json}

    Ensure that:
    - The persons are referred to as "Agent" and "Client".
    - The client name is included in the callback note, if possible.
    - The content is in German with correct grammar and spelling.
    - The output is a valid JSON object that strictly follows the given JSON structure.
    Return only the corrected JSON object with no additional text or comments.
    """

validation_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
