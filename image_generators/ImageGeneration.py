#!/usr/bin/env python3
"""
Batch image generation script using OpenAI's image generation API.

Loads prompts from a CSV file, generates images using OpenAI,
decodes the returned base64 images, and saves them locally.
"""

import argparse
from openai import OpenAI
import csv
import os
from dotenv import load_dotenv
import base64
import argparse

<<<<<<< HEAD

def main(args):
    # load OPENAI_API_KEY from environment
    load_dotenv()        

    # initialize OpenAI client                  
    client = OpenAI()   

    # create output directory if missing
    os.makedirs(args.output_dir, exist_ok=True) 

    with open(args.csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            # extract prompt text from CSV
            prompt = row["prompt"]              

            # generated image filename
            file_name = f"{args.prefix}_{i}.png" 

            try:
                response = client.images.generate(
                    model=args.model,           # OpenAI image generation model
                    prompt=prompt,              # text prompt
                    size=args.size              # output image resolution
                )

                if not response.data or not response.data[0].b64_json:
                    print(f"[{i}] ERROR: No image returned.")
                    continue

                # base64-encoded image
                image_base64 = response.data[0].b64_json  
                
                # decode to raw bytes
                image_bytes = base64.b64decode(image_base64)  

                image_path = os.path.join(args.output_dir, file_name)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)        # save image to disk

                print(f"[{i}] Saved: {image_path} | Prompt: {prompt[:50]}...")

            except Exception as e:
                print(f"Error generating image for '{prompt}': {e}")  # handle API or IO errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image generation from CSV prompts using OpenAI"
    )

    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to CSV file containing prompts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-image-1",
        help="OpenAI image generation model"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Generated image resolution"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="wine",
        help="Filename prefix for generated images"
    )

    args = parser.parse_args()
    main(args)
=======
load_dotenv()

client = OpenAI()


def generator(input_csv, output_directory, api_key):
    load_dotenv()

    client = OpenAI(api_key=api_key)

    os.makedirs(output_directory, exist_ok=True)

    with open(input_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            
            prompt = row['prompt']
            file_name = f"wine_{i}.png"
            try:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size="1024x1024"
                )
                if not response.data or not response.data[0].b64_json:
                    print(f"[{i}] ERROR: No image returned.")
                    row['file_name'] = ""
                    continue
                image_base64 = response.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)

                image_path = os.path.join(output_directory, f"rubik_{i+48}.png")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                row['file_name'] = file_name
                print(f"[{i}] Saved: {image_path} | Prompt: {prompt[:50]}...")

            except Exception as e:
                print(f"Error generating image for '{prompt}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Image Generation Script")
    parser.add_argument("--prompt_file", type=str, help="Path to csv file of prompts to be generated")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory for the images to be output to")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API key for image generation")
    args = parser.parse_args()
    generator(args.prompt_file, args.output_dir, args.api_key)

if __name__ == "__main__":
    main()
>>>>>>> 2c2c6973b907c9886f297bfc609bd047738f9241
