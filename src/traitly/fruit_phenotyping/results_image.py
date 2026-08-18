# traitly/fruit_phenotyping/results_image.py
"""
Results container and output utilities for fruit phenotyping pipelines.

Provides :class:`ResultsImage`, which stores the annotated image and
morphology and color result tables produced by
:class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`,
and exposes methods to save them to disk.
"""

# ============================================================================
# STANDARD LIBRARY
# ===========================================================================
import os
from typing import Any, Dict, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import pandas as pd

from traitly.utils.save_results import (
    _save_df,
    _ensure_dir_exists,
    _save_img,
    _to_df,
    _save_results,
    _format_output_path

)

class ResultsImage:
    """
    Container for annotated images and analysis results.

    Stores the RGB-converted annotated image alongside morphology and
    color result tables, and provides methods to save them to disk.
    Created and populated by
    :func:`~traitly.fruit_phenotyping.analysis.analyze_fruits_morphology`
    and :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.analyze_color`.

    Parameters
    ----------
    res_img : np.ndarray
        Annotated image in BGR format. Converted to RGB internally for
        display compatibility.
    morphology_results : list, optional
        List of per-fruit morphology result dictionaries. Default is an
        empty list.
    color_results : list, optional
        List of per-fruit color result dictionaries. Default is an empty
        list.
    path : str or None, optional
        Filesystem path to the original source image, used to derive
        default output paths in save methods. Default is ``None``.
    processing_metadata : dict or None, optional
        Arbitrary metadata dictionary stored for reporting purposes.
        Default is ``None``.
    """

    def __init__(
        self,
        res_img: Optional[np.ndarray],
        morphology_results: list = None,
        color_results: list = None,
        path: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ):
        if res_img is None:
            raise ValueError("res_img cannot be None when creating a ResultsImage instance.")

        self.morphology_image = res_img
        self.color_image = res_img.copy()
        self.morphology_results = morphology_results if morphology_results else []
        self.path = path
        self.color_results = color_results if color_results else []
        self._img_to_save = None

    def _resolve_img_to_save(
        self,
        image_type: str = 'auto'
    ):
        # First, check there is an image to save
        if self.morphology_image is None and self.color_image is None:
            raise RuntimeError("No image available to save.")

        if image_type == 'morphology':
            self._img_to_save = self.morphology_image
        elif image_type == 'color':
            self._img_to_save = self.color_image
        elif image_type == 'auto':
            # Check for no empty results (could be list or df)
            has_morph = (self.morphology_results is not None and
                         len(self.morphology_results) > 0)
            has_color = (self.color_results is not None and
                         len(self.color_results) > 0)

            if has_morph and self.morphology_image is not None:
                self._img_to_save = self.morphology_image
            elif has_color and self.color_image is not None:
                self._img_to_save = self.color_image
            else:
                # Fallback to whichever image exists
                self._img_to_save = self.morphology_image if self.morphology_image is not None else self.color_image
        else:
            raise ValueError("image_type must be 'morphology', 'color', or 'auto'")

    def save_img(
        self,
        output_path: Optional[str] = None,
        format: Optional[str] = None,
        output_message: bool = True,
        quality: int = 95,
        base_name: Optional[str] = None,
        image_type: str = 'auto',
        **kwargs,
    ) -> None:
        # Decide which image to save
        self._resolve_img_to_save(image_type = image_type)

        _output_path, _base_name = _format_output_path(
                                self.path,
                                base_name = base_name,
                                suffix = "_processed",
                                output_path = output_path
                            )

        _save_img(
            img=self._img_to_save,
            path=self.path,
            output_path=_output_path,
            format=format,
            verbose=output_message,
            quality=quality,
            base_name=_base_name,
        )

    def save_all(
        self,
        base_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        format: str = "jpg",
        sep: str = ",",
        output_message: bool = True,
        quality: int = 95,
        image_type: str = 'auto',
    ) -> None:
        """
        Save the annotated image, morphology CSV, and color CSV in one call.

        Derives default output paths from :attr:`path` when
        ``base_name`` or ``output_dir`` are not provided. CSV files are only
        written when the corresponding result table is non-empty.

        Parameters
        ----------
        base_name : str or None, optional
            Stem used for all output filenames. If ``None``, derived from
            :attr:`path`. Default is ``None``.
        output_dir : str or None, optional
            Directory where all files are saved. If ``None``, derived from
            :attr:`path`. Default is ``None``.
        format : str, optional
            Image format extension for the annotated image. Default is
            ``'jpg'``.
        sep : str, optional
            Column separator for CSV files. Default is ``','``.
        output_message : bool, optional
            If True, print each saved file path. Default is True.
        quality : int, optional
            JPEG compression quality in [0, 100]. Default is 95.

        Raises
        ------
        ValueError
            If ``base_name`` or ``output_dir`` cannot be determined because
            :attr:`path` is not set.
        RuntimeError
            If any file cannot be saved due to an unexpected error.
        """
        try:
            # ensure result is always in lower characters
            fmt = format.lower()

            ## 1. Save only one annotated image ##################################
            # Decide which image to save
            self._resolve_img_to_save(image_type = image_type)
            ## annotated image
            out_dir, name = _format_output_path(
                input_path=self.path,
                base_name=base_name,
                suffix=f"_processed.{fmt}",
                output_path=output_dir,
            )

            _save_img(
                img=self._img_to_save,
                path=self.path,
                output_path=os.path.join(out_dir, name),
                format=fmt,
                verbose=output_message,
                quality=quality,
            )

            # 2. For the morphology results: ##############################################

            ## CSV
            out_dir, name = _format_output_path(
                input_path=self.path,
                base_name=base_name,
                suffix="_morphology_results.csv",
                output_path=output_dir,
            )

            morph_df = self.morphology_results if isinstance(self.morphology_results, pd.DataFrame) else pd.DataFrame(self.morphology_results or {})
            _save_df(
                morph_df,
                os.path.join(out_dir, name),
                base_name="Morphology",
                sep=sep,
                verbose=output_message)

            # 3. For the color results: ##############################################
            out_dir, name = _format_output_path(
                input_path=self.path,
                base_name=base_name,
                suffix="_color_results.csv",
                output_path=output_dir,
            )

            # csv
            color_df = self.color_results if isinstance(self.color_results, pd.DataFrame) else pd.DataFrame(self.color_results or {})
            _save_df(
                color_df,
                os.path.join(out_dir, name),
                base_name="Color",
                sep=sep,
                verbose=output_message)

        except Exception as e:
            raise RuntimeError(f"> Error in save_all: {str(e)}")

    def _get_base_path(
        self,
        output_path: str,
        base_name: str
    ):
        if output_path is None:
            if not self.path:
                raise ValueError(
                    "No path provided and no original image reference available"
                )
            out_dir = os.path.dirname(self.path)
            base_path = os.path.join(out_dir, base_name)
        else:
            expanded = os.path.abspath(os.path.expanduser(output_path))

            # If user passed a directory, save inside it
            if os.path.isdir(expanded) or expanded.endswith(os.sep):
                base_path = os.path.join(expanded.rstrip(os.sep), base_name)
                _ensure_dir_exists(base_path)

            else:
                # Treat as a file path (must end with .csv)
                full_path = _ensure_dir_exists(expanded)
                stem, ext = os.path.splitext(full_path)
                if ext and ext.lower() != ".csv":
                    raise ValueError(
                        "save_csv path must end with .csv (or be a directory)"
                    )
                base_path = stem  # remove .csv so we can add suffixes consistently

        return base_path

    def save_csv(
        self,
        output_path: Optional[str] = None,
        sep: str = ",",
        output_message: bool = True,
        data: str = "auto",
        base_name: Optional[str] = None,
    ) -> None:
        """
        Save morphology results, color results, or both to CSV files.

        Resolves output paths from ``path``, ``base_name``, and
        :attr:`path`. Supports three path modes:

        - ``path=None`` — saves next to :attr:`path` using
        ``base_name`` as the filename stem.
        - ``path`` is a directory — saves inside that directory using
        ``base_name``.
        - ``path`` ends with ``.csv`` — treated as the filename stem
        (suffix ``_morphology_results.csv`` or ``_color_results.csv``
        is appended as needed).

        Parameters
        ----------
        path : str or None, optional
            Output path. See above for resolution logic. Default is ``None``.
        sep : str, optional
            Column separator for CSV output. Default is ``','``.
        output_message : bool, optional
            If True, print each saved file path. Default is True.
        data : str, optional
            Which results to save:

            - ``'auto'`` – saves morphology if available, then color if
            available. Raises if neither is available.
            - ``'morphology'`` – saves :attr:`morphology_results` only.
            - ``'color'`` – saves :attr:`color_results` only.
            - ``'both'`` – saves both; raises if neither is available.

            Default is ``'auto'``.
        base_name : str or None, optional
            Filename stem used when ``path`` is ``None`` or a directory. If
            ``None``, derived from :attr:`path`, or ``'results'`` if
            :attr:`path` is not set. Default is ``None``.

        Raises
        ------
        ValueError
            If the requested result table is empty, ``path`` is ``None`` and
            :attr:`path` is not set, ``path`` has a non-CSV extension,
            or ``data`` is not one of the supported modes.
        """
        # resolve the base name before saving the table
        if base_name is None:
                if self.path:
                    base_name = os.path.splitext(os.path.basename(self.path))[0]
                else:
                    base_name = "results"

        morph_df = _to_df(getattr(self, "morphology_results", None))
        color_df = _to_df(getattr(self, "color_results", None))

        # Get the base path (no suffix)
        base_path = self._get_base_path(output_path, base_name)

        # Decide which files are going to be saved
        mode = (data or "auto").strip().lower()

        if mode == "morphology":
            _save_results(morph_df, pd.DataFrame(), base_path, sep, output_message, require_morph=True)
        elif mode == "color":
            _save_results(pd.DataFrame(), color_df, base_path, sep, output_message, require_color=True)
        elif mode in ("both", "auto"):
            _save_results(morph_df, color_df, base_path, sep, output_message)
        else:
            raise ValueError("Data must be one of: 'auto', 'morphology', 'color', 'both'.")
