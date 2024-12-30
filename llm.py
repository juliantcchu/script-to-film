from typing import Optional
from openai import OpenAI
import json


client = OpenAI()


def get_llm_response(prompt: str, image_url: Optional[str] = None, temperature: float = 0, json_output: bool = False, show_response: bool = False) -> str:
    """
    Get a response from OpenAI's API using either text-only or text+image input.
    
    Args:
        prompt (str): The text prompt to send to the API
        image_url (Optional[str]): URL of an image to include with the prompt (for GPT-4V)
        
    Returns:
        str: The model's response text
    """
    try:
        if image_url:
            response = client.chat.completions.create(

                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": image_url}
                        ]
                    }
                ],
                temperature=temperature,
                response_format={"type": "json_object"} if json_output else None
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"} if json_output else None
            )

        if show_response:
            print(response.choices[0].message.content)

        if json_output:
            return json.loads(response.choices[0].message.content)
            
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"
