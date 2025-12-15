#!/usr/bin/env python3
"""
Command-line script for image-to-image membership inference using LPIPS.

Given a seed image and a text prompt, this script:
1. Generates multiple Img2Img outputs at different noise strengths
2. Computes LPIPS distances between the seed image and generated outputs
3. Uses the minimum LPIPS per strength as a feature vector
4. Applies a threshold-based membership inference decision
"""

import argparse
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import lpips
import torchvision.transforms as T


def main(args):
    # Device selection: use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess the seed image used for Img2Img inference
    seed = Image.open(args.seed_image).convert("RGB")

    # Initialize LPIPS perceptual similarity model (VGG backbone)
    # LPIPS measures perceptual similarity rather than pixel distance
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # Image preprocessing pipeline for LPIPS
    # Images are resized to a fixed resolution and converted to tensors
    to_tensor = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # Helper function to compute LPIPS distance between two images
    def lpips_dist(img1, img2):
        t1 = to_tensor(img1).unsqueeze(0).to(device)
        t2 = to_tensor(img2).unsqueeze(0).to(device)
        return loss_fn(t1, t2).item()

    # Load Stable Diffusion Img2Img pipeline
    # Uses fp16 on GPU for efficiency, fp32 otherwise
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Generate images for each noise strength
    strengths = args.strengths
    all_outputs = {s: [] for s in strengths}

    for s in strengths:
        print(f"Generating images (strength={s})")
        for _ in range(args.samples):
            img = pipe(
                prompt=args.prompt,               # text conditioning
                image=seed,                       # seed image
                strength=s,                       # noise injection level
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps
            ).images[0]
            all_outputs[s].append(img)

    # Compute feature vector:
    # For each strength, keep the minimum LPIPS distance
    # (closest reconstruction of the seed image)
    feature_vector = []
    for s in strengths:
        print(f"Computing LPIPS (strength={s})")
        dists = [lpips_dist(seed, img) for img in all_outputs[s]]
        feature_vector.append(min(dists))

    # Membership inference decision
    # Lower LPIPS at high noise → suspicious memorization
    print("\nFeature vector:", feature_vector)

    if feature_vector[-1] < args.threshold:
        print("Likely IN-TRAINING")
    else:
        print("Likely OUT-OF-TRAINING")


# Command-line interface definition
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Img2Img Membership Inference via LPIPS"
    )

    parser.add_argument(
        "--seed-image",
        type=str,
        required=True,
        help="Path to seed image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a glass bottle of wine filled to the brim",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.02, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Noise strengths for Img2Img"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of generated samples per strength"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="LPIPS threshold for membership inference"
    )

    args = parser.parse_args()
    main(args)
