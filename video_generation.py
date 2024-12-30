import replicate
import base64
import concurrent.futures
from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def process_shot(shot, scene_dir, i):
    """Helper function to process a single shot"""
    character = shot['character']
    actions = shot['actions']
    
    # Load character image
    img_path = Path(scene_dir) / "character_images" / f"{character}.webp"
    with open(img_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode()
    data_url = f"data:image/webp;base64,{b64_image}"
    
    input = {
        "prompt": actions,
        "first_frame_image": data_url
    }

    output = replicate.run(
        "minimax/video-01-live",
        input=input
    )
    
    # Save video with shot numberc
    output_path = Path(scene_dir) / "generated_videos" / f"shot_{i:03d}.mp4"
    with open(output_path, "wb") as file:
        file.write(output.read())
        print(f"Generated video saved to {output_path}")



def generate_scene_videos(scene, working_dir):
    """
    Generates videos for multiple scenes from storyboards and character images.
    
    Args:
        scenes (list): List of scene names/numbers to process
        working_dir (str): Base working directory containing scene subdirectories
    """
    print(f"\nProcessing scene {scene}...")
    
    # Set up paths
    scene_dir = Path(working_dir) / "scenes" / str(scene)
    output_dir = scene_dir / "generated_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load storyboard
    storyboard_path = scene_dir / "storyboard.csv"
    storyboard = pd.read_csv(storyboard_path)
    storyboard = storyboard.to_dict('records')
    # Process shots in batches of 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, shot in enumerate(storyboard):
            future = executor.submit(process_shot, shot, scene_dir, i)
            futures.append(future)
            
            # When we have 4 futures or this is the last shot, wait for completion
            if len(futures) == 4 or i == len(storyboard) - 1:
                concurrent.futures.wait(futures)
                futures = []
                
    print(f"Completed processing scene {scene}")

# Example usage:
# scenes_to_process = ["scene1", "scene2", "scene3"]
# generate_scene_videos(scenes_to_process, "project_directory")
