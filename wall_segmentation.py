import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation
)


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)


def detect_walls(image):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    original_size = image.size

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False
    )

    segmentation = logits.argmax(
        dim=1
    )[0].cpu().numpy()

    wall_id = None

    for key, value in model.config.id2label.items():

        if value.lower() == "wall":

            wall_id = int(key)
            break

    if wall_id is None:
        raise ValueError(
            "Wall class not found"
        )

    wall_mask = (
        segmentation == wall_id
    ).astype(np.uint8) * 255

    return wall_mask