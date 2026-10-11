"""Frozen offline evaluator models; deterministic transforms and FP32 features."""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from src.paths import BASE


class Evaluators:
    def __init__(self, study, device="cpu"):
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(BASE / study["dino_source"]))
        from dinov2.hub.backbones import dinov2_vitb14

        self.device = torch.device(device)
        self.dino = dinov2_vitb14(pretrained=False)
        self.dino.load_state_dict(
            torch.load(BASE / study["dino_weights"], map_location="cpu", weights_only=True),
            strict=True,
        )
        self.classifier = resnet50(weights=None)
        self.classifier.load_state_dict(
            torch.load(BASE / study["classifier_weights"], map_location="cpu", weights_only=True),
            strict=True,
        )
        for model in (self.dino, self.classifier):
            model.to(self.device).eval().requires_grad_(False)
        self.dino_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.classifier_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    @torch.inference_mode()
    def extract(self, paths, batch_size=16):
        features, probabilities = [], []
        for start in range(0, len(paths), batch_size):
            dino_input, classifier_input = [], []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    dino_input.append(self.dino_transform(rgb))
                    classifier_input.append(self.classifier_transform(rgb))
            feature = self.dino(torch.stack(dino_input).to(self.device)).float().cpu().numpy()
            probability = (
                self.classifier(torch.stack(classifier_input).to(self.device))
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )
            if not np.isfinite(feature).all() or not np.isfinite(probability).all():
                raise FloatingPointError("Nonfinite evaluator outputs")
            features.append(feature)
            probabilities.append(probability)
        return np.concatenate(features), np.concatenate(probabilities)

    def metadata(self, study):
        return dict(
            dino="dinov2_vitb14",
            dino_sha256=study["dino_sha256"],
            dino_source_commit=study["dino_source_commit"],
            classifier="ResNet50_IMAGENET1K_V2",
            classifier_sha256=study["classifier_sha256"],
            precision="fp32",
            device=str(self.device),
            torch=torch.__version__,
            dino_transform="resize256_bicubic_center224_imagenet_normalization",
            classifier_transform="torchvision_ResNet50_IMAGENET1K_V2.transforms",
            kid_features="raw final normalized CLS token, 768 dimensions",
            diversity="pairwise cosine distance on unit-normalized CLS features",
        )
