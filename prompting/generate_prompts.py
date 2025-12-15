#!/usr/bin/env python3
import csv
import prompting.gpt as gpt
import argparse

def main(args):
    base_prompt = args.base_prompt
    system_prompt = args.system_prompt
    id = 0
    combinations = []
    headers = ["prompt_id", "word_count", "descriptor_words", "num_visual_attributes", "prompt"]
    for i in range(15):
        for des_words in range(1,args.descriptor_words + 1):
            for visual in range(1, args.visual_attributes + 1):
                id += 1
                combinations.append([id, None, des_words, visual])
    
    with open(f"./{args.prompt_file}", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in combinations:
            prompt = gpt.prompt_generation(base_prompt, row, system_prompt)
            assert prompt is not None
            row[1] = len(prompt.split())
            writer.writerow(row + [prompt])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image generation from CSV prompts using OpenAI"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Path to the csv file containing prompts and variable information"
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="system_prompt.txt",
        help="Path to txt file containing the system prompt for prompt generation"
    )

    parser.add_argument(
        "--base_prompt",
        type=str,
        default="system_prompt.txt",
        help="Path to txt file containing the base prompt for prompt generation"
    )

    parser.add_argument(
        "--descriptor_words",
        type=int,
        default=4,
        help="Amount of descriptor words to stratify on for a given prompt"
    )

    parser.add_argument(
        "--visual_attributes",
        type=int,
        default=3,
        help="Amount of visual attribute words to stratify on for a given prompt"
    )
    
    parser.add_argument(
        "--api_key",
        required=True,
        type=str,
        help="API key for the OpenAI api"
    )

    args = parser.parse_args()
    main(args)
    