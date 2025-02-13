import os
from typing import List, Dict
from langchain_groq import ChatGroq

# Set Groq API Key (consider using environment variables instead of hardcoding)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

DEFAULT_SYSTEM_CONTENT = """
You are EthicALL, a smart, empathetic, and highly efficient AI designed to monitor communications
for ethical and regulatory compliance. Your job is to ensure that all messages align with organizational standards, 
legal requirements, and ethical guidelines—without being intrusive or overbearing.
You are not a watchdog; you are a guardian.
You are Strictly prohibited to answer any other irrelevant questions like "What is a Pyramid?"etc etc. Do only what is your profession.
Your purpose is to protect the organization and its people while maintaining a positive, collaborative environment. 
Let’s keep communications ethical, compliant, and professional—together.
When you include markdown text, convert them to Slack compatible ones.
When a prompt has Slack's special syntax like <@USER_ID> or <#CHANNEL_ID>, you must keep them as-is in your response.
"""

def call_llm(
    messages_in_thread: List[Dict[str, str]],
    system_content: str = DEFAULT_SYSTEM_CONTENT,
) -> str:
    """
    Call the Groq LLM with the given messages and system content.

    Args:
        messages_in_thread: List of message dictionaries with 'role' and 'content' keys
        system_content: System message content to set context for the LLM

    Returns:
        str: Raw response from the LLM without any markdown conversion
    """
    if not isinstance(messages_in_thread, list):
        raise TypeError("messages_in_thread must be a list of dictionaries")

    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

    messages = [{"role": "system", "content": system_content}]
    messages.extend(messages_in_thread)

    try:
        response = llm.invoke(messages)
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return str(response)
    except Exception as e:
        raise RuntimeError(f"Error calling Groq LLM: {str(e)}")