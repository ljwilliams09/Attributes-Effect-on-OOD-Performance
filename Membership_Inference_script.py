import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import lpips
import torchvision.transforms as T
import numpy as np

def main():
    seed_path = "/images/wine/full_0001.png"      # path to the seed image used for img2img inference
    seed_path                                     # display the path in notebook output


    loss_fn = lpips.LPIPS(net='vgg').to("cuda")   # initialize LPIPS perceptual metric and move it to GPU


    to_tensor = T.Compose([                       # preprocessing pipeline for LPIPS comparison:
        T.Resize((256,256)),                      # resize image to stable shape
        T.ToTensor()                              # convert to PyTorch tensor
    ])


    def lpips_dist(img1, img2):                   # function to compute LPIPS distance between two images
        t1 = to_tensor(img1).unsqueeze(0).to("cuda")  # preprocess and move first image to GPU
        t2 = to_tensor(img2).unsqueeze(0).to("cuda")  # preprocess and move second image to GPU
        d = loss_fn(t1, t2)                       # compute LPIPS distance
        return d.item()                           # return as scalar Python float


    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",         # Stable Diffusion model used
        torch_dtype=torch.float16                 # use half precision for speed
    ).to("cuda")                                   # load model onto GPU


    seed = Image.open(seed_path).convert("RGB")   # open seed image and ensure RGB format

    caption = "a glass bottle of wine filled to the brim"   # text prompt for Img2Img
    seed                                         # show seed image in notebook


    strengths = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]   # image noise strengths for membership inference test
    num_samples = 4                               # generate multiple samples per strength

    all_outputs = {s: [] for s in strengths}      # dictionary to collect outputs for each strength


    for s in strengths:                           # loop through all noise strengths
        print(f"Generating for strength = {s}")   # log current strength level
        for _ in range(num_samples):              # generate N samples each time
            img = pipe(
                prompt=caption,                   # text conditioning
                image=seed,                       # original seed image
                strength=s,                       # noise amount
                guidance_scale=7.5,               # prompt adherence
                num_inference_steps=50            # diffusion steps
            ).images[0]                           # extract generated image
            all_outputs[s].append(img)            # store image in results dict


    feature_vector = []                           # will hold minimum LPIPS per strength


    for s in strengths:                           # loop again for LPIPS calculation
        print(f"Computing LPIPS at strength = {s}")  # log current step
        dists = [lpips_dist(seed, img) for img in all_outputs[s]]  # compute LPIPS for all samples
        feature_vector.append(min(dists))          # keep the closest (lowest LPIPS)


    print("\nFeature vector:", feature_vector)    # print final LPIPS results


    if feature_vector[-1] < 0.25:                 # threshold comparison for membership inference
        print("Likely IN-TRAINING")               # model reproduced image too well → suspicious
    else:
        print("Likely OUT-OF-TRAINING")           # model could not reproduce → normal behavior


main()