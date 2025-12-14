import csv
import gpt_prompter

def main():
    base_prompt = "a rubik cube with just the corner missing, not an edge"
    system_prompt = "You are a research assistant that is creating prompts for us to test on out of-distribution image generation. You will be given a list of prompt attributes, and a target object for the image generator to create. Return one image prompt that follows the attribute guidelines to elicit the photo from the image generator. Return only the final image prompt with no explanation or extra text. Your main focus should getting a corner to be missing on the rubik cube, not anything else." \
    "Here are descriptions of what each variable means: " \
    "descriptor_words_count: adjectives + adverbs count" \
    "num_visual_attributes: color words + size words" \
    "style: photorealistic" 

    id = 0
    descriptor_words = 4
    visual_attributes = 3
    combinations = []
    headers = ["prompt_id", "word_count", "descriptor_words", "num_visual_attributes", "prompt"]
    for i in range(15):
        for des_words in range(1,descriptor_words + 1):
            for visual in range(1, visual_attributes + 1):
                id += 1
                combinations.append([id, None, des_words, visual])
    
    with open("./rubik_prompts.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in combinations:
            prompt = gpt_prompter.prompt_generation(base_prompt,row,system_prompt)
            assert prompt is not None
            row[1] = len(prompt.split())
            writer.writerow(row + [prompt])

if __name__ == "__main__":
    main()