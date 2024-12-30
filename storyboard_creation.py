import re
import json
import pandas as pd
from llm import get_llm_response

def script_to_storyboard(script):
    """
    Converts a script into a storyboard DataFrame with character actions and speech.
    
    Args:
        script (str): The script text to analyze
        
    Returns:
        pd.DataFrame: DataFrame containing storyboard with columns for character, actions, 
                     speaking_character and speech
    """
    # Get LLM analysis of script
    prompt = f"""Convert this script into a JSON array of objects. Each object should have:
- "character": The character in the scene (use their codename)
- "actions": What they are doing in the scene
- "speaking_character": Who is speaking (use their codename, or empty if no speech)
- "speech": What is being said (or empty if no speech)

Script:
{script}

Return only the JSON array.
    """
    
    response = get_llm_response(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        analysis = json.loads(json_str)
    else:
        raise ValueError("No JSON found in the text")

    # Convert to DataFrame
    analysis = pd.DataFrame(analysis)
    
    # Add talking to actions and prepend "the character" 
    mask = analysis['speech'].notna() & analysis['speech'].ne('')
    mask_speaking = analysis['speaking_character'] == analysis['character']
    
    # Update actions for speaking characters
    analysis.loc[mask & mask_speaking & analysis['actions'].ne('') & analysis['actions'].ne('character '), 'actions'] = \
        'the character ' + analysis.loc[mask & mask_speaking & analysis['actions'].ne('') & analysis['actions'].ne('character '), 'actions'].str.rstrip('.') + \
        ', talking to the viewer'
    analysis.loc[mask & mask_speaking & analysis['actions'].eq(''), 'actions'] = 'the character talking to the viewer'
    analysis.loc[mask & analysis['actions'].eq('character '), 'actions'] = 'the character talking'
    
    return analysis
