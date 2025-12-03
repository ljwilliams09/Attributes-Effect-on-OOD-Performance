from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

def prompt_generation(prompt, guidelines, system_prompt):
    # Guidelines
    des_words = guidelines[2]
    spatial_con = guidelines[5]
    vis_att = guidelines[4]
    amb = guidelines[5]
    sen_con = guidelines[3]

    key= os.getenv("PERSONAL_API_KEY")
    client = OpenAI(api_key=key)
    prompt = f"Prompt Guidelines:" \
    f"descriptor_words_count: {des_words}" \
    f"spatial_constraints: {spatial_con}" \
    f"num_visual_attributes: {vis_att}" \
    f"sentence_count: {sen_con}" \
    f"ambiguity: {amb}" \
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
        ]
    )

    return response.output_text
