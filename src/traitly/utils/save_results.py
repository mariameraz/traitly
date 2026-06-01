import os
from typing import Optional, Tuple

import cv2
import pandas as pd
import numpy as np

from traitly.utils.basic_functions import detect_img_name

def _ensure_dir_exists(path: str) -> str:
    """
    Ensure the parent directory of ``path`` exists and return its absolute path.

    Uses an internal cache (``_dir_cache``) to avoid redundant filesystem
    checks across repeated calls with the same directory.

    Parameters
    ----------
    path : str
        File path whose parent directory should be created if absent.
        Supports ``~`` expansion.

    Returns
    -------
    str
        Absolute version of ``path`` with its parent directory guaranteed
        to exist.
    """

    abs_path = os.path.abspath(os.path.expanduser(path))
    dir_path = os.path.dirname(abs_path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return abs_path

def _save_df(
    df: pd.DataFrame,
    output_path: str,
    base_name: str,
    sep: str = ",",
    verbose: bool = True
) -> bool:
    """Save df if not empty. Returns True if saved."""

    if df is None or df.empty:
        return False

    if not output_path.lower().endswith(".csv"):
        output_path += ".csv"

    output_path = _ensure_dir_exists(output_path)
    df.to_csv(output_path, sep=sep, index=False, encoding="utf-8", na_rep="NaN")

    if verbose:
        print(f"– {base_name} CSV saved at: {output_path}")

    return True


def _save_img(
    img: np.ndarray,
    path: Optional[str],
    output_path: Optional[str] = None,
    format: Optional[str] = None,
    verbose: bool = True,
    quality: int = 95,
    base_name: Optional[str] = None,
) -> None:
    try:
        if output_path is None or os.path.isdir(str(output_path)):
            if not path:
                raise ValueError(
                    "No path provided and no original image reference available"
                )
            if base_name is None:
                base_name = os.path.splitext(os.path.basename(path))[0]
            ext = format.lower() if format else "jpg"
            out_dir = output_path if output_path is not None else os.path.dirname(path)
            output_path = os.path.join(out_dir, f"{base_name}.{ext}")

        full_path = _ensure_dir_exists(output_path)
        format = format or os.path.splitext(full_path)[1][1:].lower()

        if format.lower() in ["jpg", "jpeg"]:
            cv2.imwrite(full_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif format.lower() == "png":
            cv2.imwrite(full_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(full_path, img)

        if verbose:
            print(f"– Image saved at: {full_path}")

    except Exception as e:
        raise RuntimeError(f"– Error saving image: {str(e)}")


def _format_output_path(
    input_path: str,
    base_name: str,
    suffix: str,
    output_path: Optional[str] = None,
) -> Tuple[str, str]:

    if output_path is None:
        output_path = os.path.dirname(input_path)

    if base_name is None:
        name = detect_img_name(input_path)
        name = os.path.splitext(name)[0]
        base_name = name + suffix
    else:
        base_name = base_name + suffix

    return output_path, base_name
