#!/usr/bin/env python3
import pandas as pd
import csv
import VQAScore
import argparse
import os
import VQAScore

# Get VQAScore
def main(args):
    filenames = []

    with open(args.input_file, newline="") as f:
        reader = csv.DictReader(f)  # pass the file object, not the filename
        for row in reader:
            filenames.append(os.path.join(args.image_dir, row["filename"]))  # append to the list, not the filename string

    # Add VQAScore to the csv
    df = pd.read_csv(args.input_file)
    df["vqascore"] = VQAScore.vqa_score(filenames, args.api_key, args.base_prompt) # type: ignore
    df.to_csv(args.input_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ground truth testing for VQAScore on a given task"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Filename for the prompt csv file to store image scores"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        help="Directory of folder containing images"
    )

    parser.add_argument(
        "--base_prompt",
        type=str,
        required=True,
        help="Base prompt for VQAScore to evaluate images against"

    )

    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for using OpenAI models with VQAScore"
    )
    args = parser.parse_args()
    main(args)