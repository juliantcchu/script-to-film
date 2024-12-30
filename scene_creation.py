from llm import get_llm_response
from generate_character_images import generate_single_image

def generate_character_shots(script, work_dir, scene, character_config):
    """
    Generates individual character shots based on a script and saves them to files.
    
    Args:
        script (str): The script text to base the scene on
        work_dir (str): Working directory for output files
        scene (str): The scene name/number for file organization
        character_config (dict): Configuration containing character details, styles and models
    """
    # Extract configs
    characters = character_config["characters"]
    style = character_config["style"]
    
    # Create mappings
    trigger_words = {char["name"]: char["trigger_word"] for char in characters.values()}
    models = {char["trigger_word"]: char.get("finetuned_model") 
             for char in characters.values() if char.get("finetuned_model")}
    char_descriptions = {char["name"]: char["description"] for char in characters.values()}

    # Get scene setup
    prompt = f"""Based on this script, give me a detailed description of:
    1. "setting": A description of the scene location. Mention all the colors in the scene.
    2. "character_shots": For each character, each containing a shot description (usually a close up or medium close up, front facing shot) of that character in their initial position and describe the background in detail. Focus mostly on the background and the objects outside of the character. Redescribe the scene in the character shot description. Focus on the color and lighting of the background and make sure it is consistent between the characters based on the scene and their position, and assume each description should be understood independently wihout looking at the other descriptions. 

    Script:
    {script}"""

    scene_setup = get_llm_response(prompt)
    print(scene_setup)

    # Convert scene setup string to JSON format
    example_json = """{
        "settings": "A description of the scene location",
        "character_shots": {
            "ihfoei": "A close up shot of ihfoei standing in the middle of the room, looking at the camera. the background is a dark blue color. The window is behind him.",
            "fiofnef": "A medium close up shot of fiofnef is standing in the corner of the room, looking at the camera. the background is a dark blue color. He is facing the window",
        }
    }"""
    prompt = f"""{scene_setup}

    Based on the above, give me a JSON object with keys settings and character_shots. Refer to each character by the following codenames: {str(trigger_words)}. 
    character_shots should be a dictionary with keys {trigger_words.values()} and values as a description of the character's shot, with a focus on the background and objects rather than the character. Refer to each character by their codename in the description as well. Describe the background in detail. Do not mention the name of the other character in the description. If the character is sitting across another character, just say that the character is sitting across the table.

    Example: 
    {example_json}

    Return the JSON object only. 
    """

    scene_setup_json = get_llm_response(prompt, json_output=True, show_response=True)
    print("\nScene Setup (JSON):")
    print(scene_setup_json)

    # Add style and character description for characters without finetuned models
    for character in scene_setup_json["character_shots"]:
        if character not in models:
            # Get the character's name from the trigger word by finding the key that maps to this value
            char_name = next(name for name, trigger in trigger_words.items() if trigger == character)
            # Get the character's description from the characters dictionary
            char_description = char_descriptions[char_name]
            scene_setup_json["character_shots"][character] = f"{style} of {scene_setup_json['character_shots'][character]}. The character is {char_description}"

    # Generate a shot for each character
    for character in scene_setup_json["character_shots"]:
        print(character, ':', scene_setup_json["character_shots"][character])
        output = generate_single_image([character], scene_setup_json["character_shots"][character], aspect_ratio="9:16", display_image=True, models=models)
        filename = f"{work_dir}/scenes/{scene}/character_shots/{character}.png"
        with open(filename, "wb") as file:
            file.write(output.read())