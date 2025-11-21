# Milestone 1

## OOD Image Generation Tasks

## Membership Inference Pipeline
Reference: Matyas Bohacek, Hany Farid; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025, pp. 321-330

This section summarizes the full pipeline used to perform membership inference on image-generation models. The goal is to determine whether a specific image was part of the model’s training data by analyzing how closely the model reconstructs it at varying noise strengths.

- 1. Load Required Packages and LPIPS Metric

We begin by loading LPIPS, a perceptual distance metric that approximates human image similarity by comparing deep VGG features instead of raw pixels.

``` python
loss_fn = lpips.LPIPS(net='vgg').to("cuda")

to_tensor = T.Compose([
    T.Resize((256,256)),   # standardize resolution
    T.ToTensor()           # convert to PyTorch tensor
])

def lpips_dist(img1, img2):
    t1 = to_tensor(img1).unsqueeze(0).to("cuda")  # add batch dimension
    t2 = to_tensor(img2).unsqueeze(0).to("cuda")
    d = loss_fn(t1, t2)                           # perceptual distance
    return d.item()                               # convert tensor → float
```

- 2. Load Stable Diffusion Img2Img Pipeline

We use the img2img pipeline since it allows injecting noise into the seed image and observing how well the model reconstructs it.

``` python
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16              # lower VRAM, faster inference
).to("cuda")                               # run all components on GPU

```

- 3. Load the Candidate Seed Image

This is the image we want to test for membership.
![Alt text](./wine.jpg)

``` python
seed = Image.open(seed_path).convert("RGB")
caption = "a glass bottle of wine filled to the brim"
seed
```

- 4. Generate Reconstructions at Multiple Strength Levels

Membership inference relies on the idea that training images tend to be reconstructed more faithfully, especially at low noise levels.

We define noise strengths and generate multiple samples per strength:

``` python
strengths = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]
num_samples = 4

all_outputs = {s: [] for s in strengths}

for s in strengths:
    print(f"Generating for strength = {s}")
    for _ in range(num_samples):
        img = pipe(
            prompt=caption,
            image=seed,
            strength=s,              # how much noise to apply
            guidance_scale=7.5,       # prompt adherence
            num_inference_steps=50
        ).images[0]

        all_outputs[s].append(img)

```
- 5. Compute LPIPS Distances for Each Strength

For each noise level, we compute perceptual similarity between the seed and generated images. We use the minimum LPIPS distance per strength, because the closest reconstruction is most indicative of membership inference.

``` python
feature_vector = []

for s in strengths:
    print(f"Computing LPIPS at strength = {s}")
    dists = [lpips_dist(seed, img) for img in all_outputs[s]]
    feature_vector.append(min(dists))

print("\nFeature vector:", feature_vector)

```

- 6. Feature Vector Interpretation

The resulting vector:

[LPIPS_strength_0.02,
 LPIPS_strength_0.2,
 LPIPS_strength_0.4,
 LPIPS_strength_0.6,
 LPIPS_strength_0.8,
 LPIPS_strength_1.0]


Forms the membership signature.

    - If the image was in training:

        Low strength → very small LPIPS (model reconstructs it well).

        Higher strength → values increase smoothly.

    - If the image was not in training:

        LPIPS stays relatively high.

        No dramatic drop at low strength.

This feature vector can then be fed into either a logistic regression classifier or a threshold-based rule for membership inference. We used a threshold-based rule that if the LIPIS is smaller than 0.25 when strength is 1, then the image is likely to be in-training. If not, then it is likely out-of-training.

The result feature vector of the wine bottle example is:
Feature vector: [0.03773997351527214, 0.0715467557311058, 0.19348135590553284, 0.17256315052509308, 0.18868671357631683, 0.32252246141433716], where we look at the last feature. We conclude (with the threshold of 0.25) that the image is out of training and is good for the target image of an ood task.

## VQAScore Working Example

### Real Image Examples and Score

- **Image generation prompt**: "A glass of red wine that is filled completely to the brim with wine."

### Prompt List

### Image Outputs

### VQAScore Comparisons
