import io
from typing import Optional, Tuple, Union, cast

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage

from .birefnet_onnx import BiRefNetSessionONNX


def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    """
    Perform a simple cutout operation on an image using a mask.
    """
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def apply_background_color(img: PILImage, color: Tuple[int, int, int, int]) -> PILImage:
    """
    Apply the specified background color to the image.
    """
    background = Image.new("RGBA", img.size, tuple(color))
    return Image.alpha_composite(background, img)


def fix_image_orientation(img: PILImage) -> PILImage:
    """
    Fix the orientation of the image based on its EXIF data.
    """
    return cast(PILImage, ImageOps.exif_transpose(img))


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    session: BiRefNetSessionONNX,
    only_mask: bool = False,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
) -> PILImage:
    """
    Remove the background from an input image.

    Parameters:
        data (Union[bytes, PILImage, np.ndarray]): The input image data.
        session (BiRefNetSessionONNX): BiRefNet session for mask generation.
        only_mask (bool): Return only the mask. Defaults to False.
        bgcolor (Optional[Tuple[int, int, int, int]]): Background color (RGBA). Defaults to None.

    Returns:
        PILImage: The cutout image with the background removed.
    """
    # Convert input to PIL Image
    if isinstance(data, bytes):
        img = cast(PILImage, Image.open(io.BytesIO(data)))
    elif isinstance(data, PILImage):
        img = data
    elif isinstance(data, np.ndarray):
        img = Image.fromarray(data)
    else:
        raise ValueError(f"Input type {type(data)} is not supported.")

    # Fix image orientation
    img = fix_image_orientation(img)

    # Get mask from BiRefNet
    masks = session.predict(img)
    mask = masks[0]

    if only_mask:
        return mask

    # Create cutout
    cutout = naive_cutout(img, mask)

    # Apply background color if specified
    if bgcolor is not None:
        cutout = apply_background_color(cutout, bgcolor)

    return cutout
