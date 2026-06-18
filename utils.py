import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

to_pil = transforms.ToPILImage()


def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(DEVICE)
    return model

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return preprocess(pil_img).unsqueeze(0).to(DEVICE, DTYPE)

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu()

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = x * std + mean
    x = torch.clamp(x, 0, 1)
    return to_pil(x)

def predict_topk(model, x, k=5):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for i, v in zip(idxs.cpu(), vals.cpu())]

def capture_activations(model, x, layers):
    activations = {}
    hooks = []

    def hook(name):
        def fn(_, __, out):
            activations[name] = out.detach()
        return fn

    for name, layer in layers.items():
        hooks.append(layer.register_forward_hook(hook(name)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return activations

def activation_mean(act):
    a = act[0].mean(dim=0)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return Image.fromarray((a.cpu().numpy() * 255).astype(np.uint8))

def activation_channel(act, ch):
    a = act[0, ch]
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return Image.fromarray((a.cpu().numpy() * 255).astype(np.uint8))

def gradcam_overlay(model, x, pil_img, layer, class_idx=None):
    cam = GradCAM(model, target_layer=layer)

    model.zero_grad(set_to_none=True)
    logits = model(x)

    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())

    cam_map = cam(class_idx, logits)[0]

    if cam_map.dim() == 3:
        cam_map = cam_map.squeeze(0)

    h, w = int(cam_map.shape[-2]), int(cam_map.shape[-1])
    base = pil_img.resize((w, h))

    out = overlay_mask(base, to_pil(cam_map.detach().cpu()), alpha=0.5)

    cam.remove_hooks()

    return out

def total_variation(x):
    return (
        torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
        + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    )

def activation_maximization(
    model,
    layer,
    channel,
    steps=200,
    lr=0.08,
    l2_weight=1e-4,
    tv_weight=1e-4,
    seed=0,
):
    torch.manual_seed(seed)
    x = torch.randn(1, 3, 224, 224, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)

    act_holder = {}

    def hook(_, __, out):
        act_holder["act"] = out

    h = layer.register_forward_hook(hook)
    history = []

    for _ in range(steps):
        opt.zero_grad()
        act_holder.clear()

        model(x)
        act = act_holder["act"]
        objective = act[:, channel].mean()

        loss = (
            -objective
            + l2_weight * (x ** 2).mean()
            + tv_weight * total_variation(x)
        )

        loss.backward()
        opt.step()
        x.data.clamp_(-3, 3)

        history.append(float(objective.detach().cpu()))

    h.remove()
    return tensor_to_pil(x), history