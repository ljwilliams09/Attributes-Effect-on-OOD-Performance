#!/usr/bin/env python3
import VQAScore.VQAScore as VQAScore
import argparse

def main(args):
    image_paths = ["./generated_images/wine/ground_truths/full_0001.png",
                "./generated_images/wine/ground_truths/full_0002.png",
                "./generated_images/wine/ground_truths/full_0003.png",
                "./generated_images/wine/ground_truths/full_0004.png",
                "./generated_images/wine/ground_truths/full_0005.png",
                "./generated_images/wine/ground_truths/half_0001.png",
                "./generated_images/wine/ground_truths/half_0002.png",
                "./generated_images/wine/ground_truths/half_0003.png",
                "./generated_images/wine/ground_truths/half_0004.png",
                "./generated_images/wine/ground_truths/half_0005.png"] 

    base_prompt = "a glass of red wine that is filled completely to the brim"
    print(f"Scores: {VQAScore.vqa_score(image_paths, args.api_key, base_prompt)}") # prints the scores from all images baesd on the base_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ground truth testing for VQAScore on a given task"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for using OpenAI models with VQAScore"
    )
    args = parser.parse_args()
    main(args)
