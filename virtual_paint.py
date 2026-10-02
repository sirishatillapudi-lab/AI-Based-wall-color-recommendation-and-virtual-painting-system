import numpy as np
import cv2
import torch

from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)


# =========================================================
# IMAGE
# =========================================================

def prepare_image(image):

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    return Image.fromarray(
        image
    ).convert("RGB")


# =========================================================
# CLASS ID
# =========================================================

def get_class_id(class_name):

    for key, value in model.config.id2label.items():

        if value.lower() == class_name.lower():
            return int(key)

    return None


# =========================================================
# GET ALL AI PREDICTIONS
# =========================================================

def get_prediction(image):

    image = prepare_image(image)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    prediction = (
        outputs.logits
        .argmax(dim=1)[0]
        .cpu()
        .numpy()
    )

    prediction = cv2.resize(
        prediction.astype(np.uint8),
        image.size,
        interpolation=cv2.INTER_NEAREST
    )

    return prediction


# =========================================================
# MASK FOR ONE CLASS
# =========================================================

def create_class_mask(
    prediction,
    class_name
):

    class_id = get_class_id(
        class_name
    )

    if class_id is None:

        return np.zeros(
            prediction.shape,
            dtype=np.uint8
        )

    return np.where(
        prediction == class_id,
        255,
        0
    ).astype(np.uint8)


# =========================================================
# CLEAN MASK
# =========================================================

def clean_mask(mask):

    kernel_small = np.ones(
        (3, 3),
        np.uint8
    )

    kernel_medium = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium
    )

    return mask


# =========================================================
# WALL MASK
# =========================================================

def detect_wall_mask_from_prediction(
    prediction
):

    wall_mask = create_class_mask(
        prediction,
        "wall"
    )

    wall_mask = clean_mask(
        wall_mask
    )

    return wall_mask


# =========================================================
# CEILING MASK
# =========================================================

def detect_ceiling_mask_from_prediction(
    prediction
):

    ceiling_mask = create_class_mask(
        prediction,
        "ceiling"
    )

    ceiling_mask = clean_mask(
        ceiling_mask
    )

    return ceiling_mask


# =========================================================
# OBJECT CLASSES
#
# These objects are NOT allowed to receive
# wall colour.
# =========================================================

PROTECTED_OBJECT_CLASSES = [

    "door",
    "window",
    "windowpane",

    "bed",
    "chair",
    "sofa",
    "table",
    "desk",

    "cabinet",
    "bookshelf",
    "shelf",
    "shelves",

    "chest of drawers",
    "counter",

    "television",
    "tv",

    "picture",
    "mirror",

    "person",

    "plant",
    "lamp",
    "vase",

    "curtain",

    "ottoman"
]


# =========================================================
# CREATE PROTECTED OBJECT MASK
# =========================================================

def detect_protected_objects(
    prediction
):

    protected_mask = np.zeros(
        prediction.shape,
        dtype=np.uint8
    )

    for class_name in (
        PROTECTED_OBJECT_CLASSES
    ):

        object_mask = create_class_mask(
            prediction,
            class_name
        )

        protected_mask = cv2.bitwise_or(
            protected_mask,
            object_mask
        )

    protected_mask = clean_mask(
        protected_mask
    )

    return protected_mask


# =========================================================
# CREATE FINAL WALL MASK
#
# Wall minus objects
# =========================================================

def create_protected_wall_mask(
    wall_mask,
    protected_objects
):

    # Remove objects from wall
    protected_wall = cv2.bitwise_and(
        wall_mask,
        cv2.bitwise_not(
            protected_objects
        )
    )

    # Small closing to repair tiny gaps
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    protected_wall = cv2.morphologyEx(
        protected_wall,
        cv2.MORPH_CLOSE,
        kernel
    )

    return protected_wall


# =========================================================
# GET MASK INFORMATION
# =========================================================

def get_mask_information(
    mask,
    surface_type
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())

    x2 = int(xs.max())
    y2 = int(ys.max())

    return {

        "mask": mask,

        "x": x1,
        "y": y1,

        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1,

        "area": int(
            np.sum(mask > 0)
        ),

        "type": surface_type
    }


# =========================================================
# DETECT WALL COMPONENTS
# =========================================================

def detect_wall_components(
    wall_mask
):

    height, width = (
        wall_mask.shape
    )

    image_area = (
        height * width
    )

    contours, _ = cv2.findContours(
        wall_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    walls = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore tiny areas
        if area < image_area * 0.015:
            continue

        current_mask = np.zeros_like(
            wall_mask
        )

        cv2.drawContours(
            current_mask,
            [contour],
            -1,
            255,
            cv2.FILLED
        )

        information = (
            get_mask_information(
                current_mask,
                "wall"
            )
        )

        if information is None:
            continue

        if information["width"] < 60:
            continue

        if information["height"] < 60:
            continue

        walls.append(
            information
        )

    return walls


# =========================================================
# CONSERVATIVE WALL CORNER DETECTION
#
# This does NOT split every wall.
#
# A wall is split only when there is a strong,
# long vertical corner-like boundary.
# =========================================================

def find_wall_corner(
    image,
    wall
):

    image_array = np.array(
        prepare_image(image)
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    mask = wall["mask"]

    x = wall["x"]
    y = wall["y"]

    w = wall["width"]
    h = wall["height"]

    if w < 180 or h < 150:
        return None

    # -----------------------------------------------------
    # Only search inside the wall
    # -----------------------------------------------------

    x_start = x + int(
        w * 0.20
    )

    x_end = x + int(
        w * 0.80
    )

    if x_end <= x_start:
        return None

    # -----------------------------------------------------
    # Vertical edge detection
    # -----------------------------------------------------

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    edge_strength = np.abs(
        sobel_x
    )

    candidates = []

    for column in range(
        x_start,
        x_end
    ):

        wall_pixels = mask[
            :,
            column
        ] > 0

        values = edge_strength[
            :,
            column
        ][wall_pixels]

        if len(values) < 20:
            continue

        score = float(
            np.mean(values)
        )

        # How much of the wall height contains
        # wall pixels around this column?
        coverage = (
            np.sum(wall_pixels)
            / max(h, 1)
        )

        if coverage < 0.30:
            continue

        candidates.append(
            (
                score,
                column
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    best_score, best_x = (
        candidates[0]
    )

    # Strong edge requirement
    if best_score < 20:
        return None

    relative_x = (
        best_x - x
    ) / max(w, 1)

    if relative_x < 0.20:
        return None

    if relative_x > 0.80:
        return None

    # -----------------------------------------------------
    # Make sure both sides contain substantial wall
    # -----------------------------------------------------

    left = mask[
        :,
        x:best_x
    ]

    right = mask[
        :,
        best_x:x + w
    ]

    left_area = np.sum(
        left > 0
    )

    right_area = np.sum(
        right > 0
    )

    total_area = (
        left_area +
        right_area
    )

    if total_area == 0:
        return None

    left_ratio = (
        left_area /
        total_area
    )

    right_ratio = (
        right_area /
        total_area
    )

    # Both walls must be reasonably large
    if left_ratio < 0.20:
        return None

    if right_ratio < 0.20:
        return None

    return best_x


# =========================================================
# SPLIT ONLY A REAL CORNER
# =========================================================

def split_real_corner(
    image,
    wall
):

    split_x = find_wall_corner(
        image,
        wall
    )

    if split_x is None:

        return [wall]

    original_mask = (
        wall["mask"]
    )

    left_mask = np.zeros_like(
        original_mask
    )

    right_mask = np.zeros_like(
        original_mask
    )

    left_mask[
        :,
        :split_x
    ] = original_mask[
        :,
        :split_x
    ]

    right_mask[
        :,
        split_x:
    ] = original_mask[
        :,
        split_x:
    ]

    left_mask = clean_mask(
        left_mask
    )

    right_mask = clean_mask(
        right_mask
    )

    left_info = get_mask_information(
        left_mask,
        "wall"
    )

    right_info = get_mask_information(
        right_mask,
        "wall"
    )

    if (
        left_info is None
        or
        right_info is None
    ):

        return [wall]

    # -----------------------------------------------------
    # Reject tiny split
    # -----------------------------------------------------

    original_area = (
        wall["area"]
    )

    if (
        left_info["area"]
        < original_area * 0.20
    ):

        return [wall]

    if (
        right_info["area"]
        < original_area * 0.20
    ):

        return [wall]

    return [
        left_info,
        right_info
    ]


# =========================================================
# DETECT WALLS
# =========================================================

def detect_walls(
    mask,
    image=None
):

    walls = detect_wall_components(
        mask
    )

    # -----------------------------------------------------
    # If image is available, look for a REAL corner.
    # -----------------------------------------------------

    if image is not None:

        improved_walls = []

        for wall in walls:

            split_result = (
                split_real_corner(
                    image,
                    wall
                )
            )

            improved_walls.extend(
                split_result
            )

        walls = improved_walls

    # -----------------------------------------------------
    # Sort left → right
    # -----------------------------------------------------

    walls.sort(
        key=lambda item: item["x"]
    )

    return walls


# =========================================================
# CEILING DETECTION
# =========================================================

def detect_ceiling_surfaces(
    ceiling_mask
):

    height, width = (
        ceiling_mask.shape
    )

    image_area = (
        height * width
    )

    contours, _ = cv2.findContours(
        ceiling_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    ceilings = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.005:
            continue

        current_mask = np.zeros_like(
            ceiling_mask
        )

        cv2.drawContours(
            current_mask,
            [contour],
            -1,
            255,
            cv2.FILLED
        )

        information = (
            get_mask_information(
                current_mask,
                "ceiling"
            )
        )

        if information is None:
            continue

        if information["width"] < 50:
            continue

        ceilings.append(
            information
        )

    ceilings.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    # Keep largest meaningful ceiling
    if len(ceilings) > 1:

        largest = ceilings[0]

        remaining = []

        for ceiling in ceilings[1:]:

            if (
                ceiling["area"]
                >=
                largest["area"] * 0.25
            ):

                remaining.append(
                    ceiling
                )

        ceilings = [
            largest
        ] + remaining

    return ceilings


# =========================================================
# DETECT ALL SURFACES
# =========================================================

def detect_surfaces(
    image
):

    image = prepare_image(
        image
    )

    # One AI prediction for everything
    prediction = get_prediction(
        image
    )

    # -----------------------------------------------------
    # WALL
    # -----------------------------------------------------

    raw_wall_mask = (
        detect_wall_mask_from_prediction(
            prediction
        )
    )

    # -----------------------------------------------------
    # OBJECTS
    # -----------------------------------------------------

    protected_objects = (
        detect_protected_objects(
            prediction
        )
    )

    # -----------------------------------------------------
    # WALL MINUS OBJECTS
    # -----------------------------------------------------

    protected_wall_mask = (
        create_protected_wall_mask(
            raw_wall_mask,
            protected_objects
        )
    )

    # -----------------------------------------------------
    # WALLS
    # -----------------------------------------------------

    walls = detect_walls(
        protected_wall_mask,
        image
    )

    # -----------------------------------------------------
    # CEILING
    # -----------------------------------------------------

    ceiling_mask = (
        detect_ceiling_mask_from_prediction(
            prediction
        )
    )

    ceilings = (
        detect_ceiling_surfaces(
            ceiling_mask
        )
    )

    # -----------------------------------------------------
    # FINAL SURFACES
    # -----------------------------------------------------

    surfaces = []

    surfaces.extend(
        ceilings
    )

    surfaces.extend(
        walls
    )

    return surfaces


# =========================================================
# VISUALIZATION
# =========================================================

def create_surface_visualization(
    image,
    surfaces
):

    image = prepare_image(
        image
    )

    visualization = np.array(
        image
    ).copy()

    wall_number = 1

    for surface in surfaces:

        mask = surface["mask"]

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Boundary
        cv2.drawContours(
            visualization,
            contours,
            -1,
            (255, 255, 255),
            3
        )

        x = surface["x"]
        y = surface["y"]

        if surface["type"] == "ceiling":

            label = "CEILING"

        else:

            label = (
                f"WALL {wall_number}"
            )

            wall_number += 1

        cv2.putText(
            visualization,
            label,
            (
                x + 10,
                max(
                    y + 35,
                    35
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    return visualization


# =========================================================
# REALISTIC VIRTUAL PAINTING
# =========================================================

def paint_detected_surfaces(
    image,
    surfaces,
    surface_colors
):

    image = prepare_image(
        image
    )

    original = np.array(
        image
    ).astype(np.float32)

    result = original.copy()

    for index, surface in enumerate(
        surfaces
    ):

        if index >= len(
            surface_colors
        ):
            break

        mask = surface["mask"]

        mask_float = (
            mask.astype(
                np.float32
            ) / 255.0
        )

        # -------------------------------------------------
        # Slightly soften mask edges
        # -------------------------------------------------

        mask_float = cv2.GaussianBlur(
            mask_float,
            (5, 5),
            0
        )

        # -------------------------------------------------
        # Selected RGB colour
        # -------------------------------------------------

        color = np.array(
            surface_colors[index],
            dtype=np.float32
        )

        # -------------------------------------------------
        # Preserve original lighting
        #
        # Calculate brightness from original image.
        # -------------------------------------------------

        brightness = np.mean(
            original,
            axis=2
        )

        brightness_mean = np.mean(
            brightness[
                mask > 0
            ]
        )

        if brightness_mean <= 0:
            brightness_mean = 128

        brightness_factor = (
            brightness
            / brightness_mean
        )

        brightness_factor = np.clip(
            brightness_factor,
            0.55,
            1.45
        )

        realistic_color = (
            color[None, None, :]
            *
            brightness_factor[
                :, :, None
            ]
        )

        realistic_color = np.clip(
            realistic_color,
            0,
            255
        )

        # -------------------------------------------------
        # Preserve some original texture
        # -------------------------------------------------

        painted = (
            original * 0.25
            +
            realistic_color * 0.75
        )

        # -------------------------------------------------
        # Apply only inside mask
        # -------------------------------------------------

        alpha = (
            mask_float[:, :, None]
            * 0.82
        )

        result = (
            result * (1 - alpha)
            +
            painted * alpha
        )

    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )

    return result


# =========================================================
# COMPATIBILITY FUNCTION
# =========================================================

def apply_surface_colors(
    image,
    surface_colors
):

    surfaces = detect_surfaces(
        image
    )

    result = paint_detected_surfaces(
        image,
        surfaces,
        surface_colors
    )

    return (
        result,
        surfaces
    )


# =========================================================
# OLD FUNCTIONS
# =========================================================

def create_wall_visualization(
    image,
    walls
):

    return create_surface_visualization(
        image,
        walls
    )


def apply_wall_colors(
    image,
    wall_colors
):

    wall_mask = detect_wall_mask(
        image
    )

    walls = detect_walls(
        wall_mask,
        image
    )

    result = paint_detected_surfaces(
        image,
        walls,
        wall_colors
    )

    return (
        result,
        wall_mask,
        walls
    )