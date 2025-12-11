from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

def prompt_generation(prompt, guidelines, system_prompt):
    # Guidelines
    des_words = guidelines[2]
    vis_att = guidelines[3]

    key= os.getenv("PERSONAL_API_KEY")
    client = OpenAI(api_key=key)
    prompt = f"Prompt Guidelines:" \
    f"descriptor_words_count: {des_words}" \
    f"num_visual_attributes: {vis_att}" \
    f"style: photorealistic"

    response = client.responses.create(
        model="gpt-5.1",
        input=[
            {
                "role":"system",
                "content": system_prompt
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0
    )

    return response.output_text
