# Attributes-Effect-on-OOD-Performance

## Overview

This project examines how prompt attributes affect a generative model’s performance on out-of-distribution tasks, which include both never-seen concepts and rare configurations of familiar objects encountered during training. We use membership inference to determine whether a task is out of distribution and apply VQA score to evaluate how closely generated images align with ground-truth descriptions. Using one generative model and three prompt settings consisting of base mid-OOD, low-OOD, and high-OOD tasks, we identify clear differences in model behavior. For mid-OOD tasks, additional visual attributes and descriptive modifiers tend to reduce performance, while increased prompt length often improves alignment. In low-OOD tasks, prompt attributes show little consistent influence on performance. For high-OOD tasks, the model struggles across all prompt designs, indicating limits in its ability to generalize to highly unfamiliar scenarios.

## Replication Instructions

In the following instructions, we use the wine prompt as an example to illustrate each step of the pipeline. All required files have already been created and saved in the artifact and can be used for verification.

1. **Membership Inference**

   To determine whether a prompt is out of distribution, run the Membership_Inference.ipynb notebook five times, each time uploading one of the five images of a full glass of wine. For each image, the notebook outputs whether the image is likely in-training or out-of-training. Record the feature vector produced for each run and compute the average of the final element across the five vectors. If this average exceeds 0.25, the prompt is classified as likely out of distribution. Higher average values indicate a greater degree of out-of-distribution behavior.

2. **VQA Ground Truth Testing**

3. **Prompt stratification**
   Run prompt_stratification.py in the prompting folder to generate the wine_prompts.csv file. This file contains prompts with systematically varied combinations of prompt attributes.

4. **Image generation**
   Run ImageGeneration.py in the image_generators folder to generate images for each stratified prompt. The resulting images are automatically saved in the generated_images directory, with wine-specific outputs stored in generated_images/wine. This step may require sufficient billing allowance to complete successfully.

5. **VQAScore Evaluation**

6. **Statistical Analysis**

## Future Directions

This pipeline is primarily constrained by the amount of available computational resources, and expanding the number of prompts and generated images would help strengthen the observed findings and support broader extrapolation across out-of-distribution scenarios. With additional scale, the pipeline could enable stronger correlations between specific prompt attributes and model performance. It could also potentially reveal clearer differences among types of out-of-distribution tasks and provide insight into whether image generation models respond similarly to prompt attributes under unfamiliar conditions. It offers a framework for comparing how different VQA evaluation models behave across tasks, allowing for a more nuanced understanding of how evaluation methods themselves influence performance assessment.

## Contributions

Matthew:

- Membership Inference Recreation + Testing 6-8 hrs
- Image Generation Script + Creating Images 6-8 hrs
- Poster Intro/Methodology/Conclusion 2 hrs
- Final Artfact Readme Writeup 2 hrs

Luca:

- VQAScore

Both:

- Prompt Idea Brainstorm
- Literature Review
- First Milestion Proposal Draft

## References

### VQAScore

- [GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation Repository](https://github.com/linzhiqiu/t2v_metrics)

### Membership Inference

- [GenAI Confessions: Black-box Membership Inference for Generative Image Models](https://arxiv.org/pdf/2501.06399)
