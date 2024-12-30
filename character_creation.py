import replicate
import os
from IPython.display import Image, display
import requests
import base64

import os
import shutil
import openai
import base64

import replicate

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def generate_character_images(character, description, style, working_dir, display_image=False):
    # Create characters directory if it doesn't exist
    characters_dir = os.path.join(working_dir, 'characters')
    if not os.path.exists(characters_dir):
        os.makedirs(characters_dir)

    try:
        print(f"Generating image for {character}...")
        
        # Create the prompt and input parameters
        input = {
            "prompt": f"{style} of {description}.",
            "prompt_upsampling": True
        }
        
        # Run Flux 1.1 Pro model
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input
        )
        
        # Save and display the generated image
        if output:
            filename = os.path.join(characters_dir, f"{character.lower()}.jpg")
            with open(filename, "wb") as file:
                file.write(output.read())
            if display_image:
                display(Image(filename=filename))
            print(f"{character} image generated and saved as {filename}\n")
        else:
            print(f"Failed to generate image for {character}\n")

    except Exception as e:
        print(f"Error generating image for {character}: {str(e)}")




def generate_character_variations(character, style, working_dir, number_of_outputs=20, number_of_images_per_pose=1, disable_safety_checker=True, display_image=False):
    """
    Generate variations of a character using an existing initial image.
    
    Args:
        character (str): Name of the character
        style (str): Style description for the variations
        working_dir (str): Working directory path
        number_of_outputs (int): Number of variations to generate
        number_of_images_per_pose (int): Number of images per pose
        disable_safety_checker (bool): Whether to disable safety checker
        display_image (bool): Whether to display generated images
    """
    try:
        # Setup character directories
        character_dir = os.path.join(working_dir, 'characters', character.lower())
        if not os.path.exists(character_dir):
            os.makedirs(character_dir)
            
        # Get initial image path
        image_path = os.path.join(working_dir, 'characters', f"{character.lower()}.jpg")
        
        if os.path.exists(image_path):
            # Encode image as base64
            with open(image_path, "rb") as file:
                data = base64.b64encode(file.read()).decode("utf-8")
                initial_image = f"data:application/octet-stream;base64,{data}"
            
            # Setup input parameters for variation generation
            input_variations = {
                "prompt": style,
                "subject": initial_image,
                "number_of_outputs": number_of_outputs,
                "number_of_images_per_pose": number_of_images_per_pose,
                "disable_safety_checker": disable_safety_checker
            }
            
            # Generate variations using the consistent-character model
            outputs = replicate.run(
                "fofr/consistent-character:9c77a3c2f884193fcee4d89645f02a0b9def9434f9e03cb98460456b831c8772",
                input=input_variations
            )
            
            # Save and optionally display each variation
            for i, output_url in enumerate(outputs):
                variation_filename = os.path.join(character_dir, f"variation_{i+1}.jpg")
                
                # Download and save the image from the URL
                response = requests.get(output_url)
                if response.status_code == 200:
                    with open(variation_filename, "wb") as file:
                        file.write(response.content)
                    if display_image:
                        display(Image(filename=variation_filename))
                    print(f"Variation {i+1} saved as {variation_filename}")
                else:
                    print(f"Failed to download variation {i+1}")
                    
            print(f"Generated variations for {character}\n")
        else:
            print(f"Initial image not found for {character}\n")
            
    except Exception as e:
        print(f"Error generating variations for {character}: {str(e)}")


def process_variations_for_finetuning(character_config, working_dir, zip_dir=True):
    """
    Process character variations for fine-tuning by copying images and generating captions.
    
    Args:
        characters: List of character names
        trigger_words: Dictionary mapping character names to their trigger words
        client: OpenAI client instance
    """

    # Load OpenAI API key from .env file
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Create finetune directory if it doesn't exist
    os.makedirs('finetune', exist_ok=True)
    
    # Copy and rename all variations with trigger words
    for character in character_config['characters']:
        if not character_config['characters'][character]['finetune']:
            print(f"Skipping {character} as finetune is False")
            continue
        trigger_word = character
        character_dir = f"{working_dir}/characters/{character.lower()}"
        
        # Check if character directory exists
        if os.path.exists(character_dir):
            # Get all variation images
            variation_files = [f for f in os.listdir(character_dir) if f.startswith('variation_') and f.endswith('.jpg')]
            
            for file in variation_files:
                # Get variation number
                variation_num = file.split('_')[1].split('.')[0]
                
                # Source and destination paths
                src_path = os.path.join(character_dir, file)
                
                # Read image and convert to base64
                with open(src_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                # Get AI caption
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                # idea: no clothing description? We want clothing to be the same with or without description afterwards
                                {"type": "text", "text": f"Describe The pose and actions, and expression of the character in this image, and refer to the character in this image as {trigger_word}. No need to mention clothing. For example, '{trigger_word} is looking to the left and smiling. If there is no particular expression, just say 'A photo of {trigger_word}'."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                # Extract caption and ensure trigger word is used
                caption = response.choices[0].message.content
                caption = caption.replace(character_config['characters'][character]['name'], trigger_word)  # Ensure character name is replaced with trigger word
                
                # Create and write caption file
                caption_path = os.path.join(character_dir, f"variation_{variation_num}.txt")
                with open(caption_path, 'w') as f:
                    f.write(caption)
                
                print(f"Processed variation {variation_num} for {character}: {caption}")

                if zip_dir:
                    # Zip the directory
                    shutil.make_archive(
                        character_dir,  # Output path/name without extension
                        'zip',                                         # Archive format
                        character_dir,                                 # Directory to zip
                        base_dir=None                                 # Directory inside the archive
                    )
                    print(f"Zipped {character_dir} to {character_dir}.zip")
        else:
            print(f"Directory not found for {character}")

    print("\nAll variations processed with corresponding caption files")



def finetune_character(working_dir, character, project_name='show'):
    """
    Creates and trains a fine-tuned model for a character using FLUX.1
    
    Args:
        working_dir: Base directory containing character subdirectories
        character: Name of character to fine-tune (must match directory name)
        
    Returns:
        str: The model ID in format "owner/model-name"
    """
    # Create the model
    model = replicate.models.create(
        owner="juliantcchu",
        name=f"flux-{character}",
        visibility="private",
        hardware="gpu-t4",
        description=f"A fine-tuned FLUX.1 model for {character} in project {project_name}"
    )

    # Start training
    character_dir = f"{working_dir}/characters/{character}.zip"
    training = replicate.trainings.create(
        version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
        input={
            "input_images": open(character_dir, "rb"),
            "steps": 1000
        },
        destination=f"{model.owner}/{model.name}"
    )

    print(f"Training started for {character}")
    print(f"Training URL: https://replicate.com/p/{training.id}")
    
    return f"{model.owner}/{model.name}"