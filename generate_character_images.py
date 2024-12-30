import replicate
from IPython.display import Image, display

def generate_single_image(characters, prompt, aspect_ratio="1:1", display_image=False, models=[], default_model_path="black-forest-labs/flux-1.1-pro"):
    output = None

    
    if len(characters) > 1:
        print("using multi lora")
        output = replicate.run(
            "lucataco/flux-dev-multi-lora:2389224e115448d9a77c07d7d45672b3f0aa45acacf1c5bcf51857ac295e3aec",
            input = {
                "prompt": prompt,
                "hf_loras": [models[char]["model_weights"] for char in characters],
                "negative_prompt": "blurry, distorted, low quality, deformed",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "disable_safety_checker": True,
                "aspect_ratio": aspect_ratio
            }
        )[0]
    
    else:
        print("using single model")
        model_path = models[characters[0]]["model_path"] if characters[0] in models else default_model_path
        output = replicate.run(
            model_path,
            input={
                "prompt": prompt,
                "negative_prompt": "blurry, distorted, low quality, deformed",
                "num_inference_steps": 50,
                "guidance_scale": 7.5, 
                "disable_safety_checker": True,
                "aspect_ratio": aspect_ratio
            }
        )

    try:
        output = output[0]
    except:
        print("output is not a list")
        output = output


    # Save and display the generated image
    if display_image:
        filename = "temp_generated_image.jpg"
        with open(filename, "wb") as file:
            file.write(output.read())
        display(Image(filename=filename))




    return output