## Pretrained Models

Use this directory to store any large, pre-downloaded model checkpoints that cannot be fetched automatically during development (for example, custom HuggingFace weights or ImageNet-pretrained backbones).

Suggested layout:

- `models/pretrained/resnet18/` – place `.pt` or `.pth` files here
- `models/pretrained/<model_name>/metadata.json` – optional notes such as source URL, expected input size, and license details

Update the training/evaluation scripts to load checkpoints from these folders when needed. The codebase will not create or download these files automatically.***

