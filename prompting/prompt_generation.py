import csv
import gpt_prompter
import argparse

def prompting(system, base):
    base_prompt = "a glass of red wine that is filled completely to the brim"
    system_prompt = 
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
    
    with open("./wine_prompts.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in combinations:
            prompt = gpt_prompter.prompt_generation(base_prompt,row,system_prompt)
            assert prompt is not None
            row[1] = len(prompt.split())
            writer.writerow(row + [prompt])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image generation from CSV prompts using OpenAI"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=True,
        help="Path to txt file containing the system prompt for prompt generation"
    )

    parser.add_argument(
        "--base_prompt",
        type=str,
        required=True,
        help="Path to txt file containing the base prompt for prompt generation"
    )

    parser.add_argument(
        "--descriptor_words",
        type=str,
        require=True,
        help="Amount of descriptor words to stratify on for a given prompt")


    main()