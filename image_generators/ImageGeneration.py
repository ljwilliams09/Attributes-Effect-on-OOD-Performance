from openai import OpenAI
import csv
import os
from dotenv import load_dotenv
import base64

load_dotenv()

client = OpenAI()

csv_file_path = './Tasks/stratification/test.csv'
output_directory = 'generated_images'

os.makedirs(output_directory, exist_ok=True)

with open(csv_file_path, mode='r', encoding='utf-8') as file:
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

            image_path = os.path.join(output_directory, f"wine_{i}.png")
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            row['file_name'] = file_name
            print(f"[{i}] Saved: {image_path} | Prompt: {prompt[:50]}...")

        except Exception as e:
            print(f"Error generating image for '{prompt}': {e}")
