# Attributes-Effect-on-OOD-Performance

## Overview

This project examines how prompt attributes affect a generative model’s performance on out-of-distribution tasks, which include both never-seen concepts and rare configurations of familiar objects encountered during training. We use membership inference to determine whether a task is out of distribution and apply VQA score to evaluate how closely generated images align with ground-truth descriptions. Using one generative model and three prompt settings consisting of base mid-OOD, low-OOD, and high-OOD tasks, we identify clear differences in model behavior. For mid-OOD tasks, additional visual attributes and descriptive modifiers tend to reduce performance, while increased prompt length often improves alignment. In low-OOD tasks, prompt attributes show little consistent influence on performance. For high-OOD tasks, the model struggles across all prompt designs, indicating limits in its ability to generalize to highly unfamiliar scenarios.

## Replication Instructions

In the following instructions, we use the wine prompt as an example to illustrate each step of the pipeline. All required files have already been created and saved in the artifact and can be used for verification.

1. **Determine an OOD task with membership inference**

   To assess whether a prompt is out of distribution (OOD), we run `membership_inference/Membership_Inference_script.py` five times, each time using a different seed image (indexed from full_0001.png to full_0005.png). The script is executed using the following command:

   ```python
   ./membership_inference/Membership_Inference_script.py \
   --seed-image ./membership_inference/images/wine/full_0001.png \
   --prompt "a glass bottle of wine filled to the brim"
   ```

   For each run, the script outputs a feature vector consisting of the minimum LPIPS distance at each noise strength, along with a binary classification indicating whether the image is likely in-training or out-of-training. We record the feature vector from each run and compute the average of the final element (corresponding to the highest noise strength) across the five vectors.

   If this average exceeds 0.25, the prompt is classified as likely out of distribution. Larger average values indicate a greater degree of out-of-distribution behavior, reflecting the model’s reduced ability to reconstruct the seed image under high noise.

2. **Ground truth testing with VQAScore**

   After we have established a task is OOD, we want to ensure that the VQAScore method is able to reliably distinguish between images that successful accomplish the task, and images that fail at the task.

   For this, we use the VQAScore in the `t2v_metrics` folder. `t2v_metrics` is a submodule of another repository and specific directions can be found [here]().

3. **Prompt stratification**
   Run "prompting/prompt_stratification.py" in the prompting folder to generate the wine_prompts.csv file. This file contains prompts with systematically varied combinations of prompt attributes.

4. Image generation
   Run image_generators/ImageGeneration.py with the stratified prompt CSV specified via the command line to generate images for each prompt. The generated images are automatically saved to the designated output directory (e.g., generated_images/, with wine-specific outputs stored under generated_images/wine/). This step may require sufficient billing allowance to complete successfully.

   ```python
   ./image_generators/ImageGeneration.py \
   --csv-file ./prompting/prompts/wine_prompts.csv \
   --output-dir generated_images/wine \
   --prefix wine
   ```

5. Obtain VQA Score

6. Statistical Analysis

## Future Directions

An important direction for future work is to address the mismatch between the target model used for membership inference and the model used for image generation during prompting. Identifying or developing a unified model that supports both text-to-image and image-to-image pipelines would enable more consistent evaluation and reduce uncertainty introduced by cross-model assumptions. Future work should also expand the scale of the pipeline by increasing the number of prompts and generated images, which is currently limited by computational resources. Greater scale would allow for stronger statistical confidence, clearer correlations between specific prompt attributes and model performance, and more reliable comparisons across different types of OOD tasks. With sufficient expansion, this framework could also be used to compare how different VQA-based evaluation models behave across tasks, offering deeper insight into how evaluation methodology itself influences performance assessment under unfamiliar conditions.

## Contributions

Matthew

- Membership inference recreation + testing: 6-8 hrs
- Image generation script + creating images: 6-8 hrs
- Poster intro/methodology/conclusion: 2 hrs

Luca

- VQAScore adaptation + testing: 8 hrs
- Prompt stratification methods: 6 hrs
- Poster background/intro/further directions/results: 4 hrs

Both

- Prompt idea brainstorm
- Literature review
- First milestone proposal draft
- Final artfact readme writeup

## Repository Guide

- `./analysis`
  - Folder containing R script to regress prompt variables against VQAScore outcome.
- `./documents`
  - Directory for miscellaneous documents from brainstorming, proposals, and milestones.
- `./generated_images`
  - Directory containing a folder for each task tested
  - Each task contains:
    - `ground_truths`: ground truth materials containing 5 true images and 5 false images as well as VQA outputs on a base prompt
    - `images`: generated images from the stratified prompt csv
    - `analysis.txt`: results from regression with R
    - `prompts.txt`: system prompts and base prompts used to stratify prompts
    - `./..._prompts.csv`: csv containing prompts, variables, and results for a given task
- `./image_generators`
  - contains `./ImageGeneration.py` which is the main script for generating images taken from prompts in a csv
- `./membership_inference`
  - Directory with materials for membership inferences script in python and notebook form.
  - Also contains `images` which is a folder with images to test membership inference on for a given task, and results from that test
- `./prompting`

  - Contains `generate_prompts.py` which is the main script for stratifying on prompts.
  - `base_prompt.txt` and `system_prompt.txt` are both txt files containing the system prompt and base prompt for the prompt generation, examples can be found in the generated images folder under a `prompt.txt` file for a given task.

- `./t2v_metrics`
  - submodule folder of the repository for the VQAScore

## References

[GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation](https://github.com/linzhiqiu/t2v_metrics)

[GenAI Confessions: Black-box Membership Inference for Generative Image Models](https://arxiv.org/abs/2501.06399)
