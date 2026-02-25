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
from typing import Optional, Dict, Any


# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import os
import pandas as pd
import numpy as np

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
    bgr_img : np.ndarray
        Annotated image in BGR format. Converted to RGB internally for
        display compatibility.
    morphology_results : list, optional
        List of per-fruit morphology result dictionaries. Default is an
        empty list.
    color_results : list, optional
        List of per-fruit color result dictionaries. Default is an empty
        list.
    image_path : str or None, optional
        Filesystem path to the original source image, used to derive
        default output paths in save methods. Default is ``None``.
    processing_metadata : dict or None, optional
        Arbitrary metadata dictionary stored for reporting purposes.
        Default is ``None``.
    """

    def __init__(self, 
                 bgr_img: np.ndarray, 
                 morphology_results: list = None, 
                 color_results: list = None, 
                 image_path: Optional[str] = None, 
                 processing_metadata: Optional[Dict[str, Any]] = None,
                 ):
        # Save both BGR (for cv2) and RGB (for display) to avoid reconversion
        self.annotated_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)    # Original BGR format
        self.morphology_results = morphology_results if morphology_results else []   
        self.table = self.morphology_results                  
        self.image_path = image_path
        self._dir_cache = {}  # Cache for directory checks
        self.color_results = color_results if color_results else []
        
        # Save metadata for reports
        self.processing_metadata = processing_metadata or {}

    def _ensure_dir_exists(self, path: str) -> str:
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
        
        # Check cache first
        if dir_path in self._dir_cache:
            return abs_path
        
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        # Cache the result
        self._dir_cache[dir_path] = True
        
        return abs_path

    def save_img(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        output_message: bool = True,
        quality: int = 95,
        **kwargs,
    ) -> None:
        """
        Save the annotated image to disk.

        Converts :attr:`annotated_image` from RGB back to BGR before writing
        via ``cv2.imwrite``. JPEG and PNG are written with explicit quality
        and compression settings; all other formats use ``cv2.imwrite``
        defaults.

        Parameters
        ----------
        path : str or None, optional
            Output file path. If ``None``, the image is saved next to
            :attr:`image_path` with ``'_annotated'`` appended to the stem.
            Default is ``None``.
        format : str or None, optional
            Image format extension (e.g. ``'jpg'``, ``'png'``). If ``None``,
            inferred from ``path``. Used to determine the default filename
            when ``path`` is ``None``. Default is ``None``.
        output_message : bool, optional
            If True, print the saved file path. Default is True.
        quality : int, optional
            JPEG compression quality in [0, 100]. Default is 95.
        **kwargs
            Ignored. Accepted for forward-compatibility.

        Raises
        ------
        ValueError
            If ``path`` is ``None`` and :attr:`image_path` is not set.
        RuntimeError
            If the image cannot be saved due to an unexpected error.
        """
        try:
            if path is None:
                if not self.image_path:
                    raise ValueError("No path provided and no original image reference available")
                
                original_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                ext = format.lower() if format else 'jpg'
                path = os.path.join(original_dir, f"{base_name}_annotated.{ext}")

            full_path = self._ensure_dir_exists(path)
            format = format or os.path.splitext(full_path)[1][1:].lower()

            bgr_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
            # Use cv2.imwrite
            if format.lower() in ['jpg', 'jpeg']:
                # JPEG with quality setting
                cv2.imwrite(full_path, bgr_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                # PNG with compression
                cv2.imwrite(full_path, bgr_image, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                # Other formats - default
                cv2.imwrite(full_path, bgr_image)
            
            if output_message:
                print(f"Image saved at: {full_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error saving image: {str(e)}")
    
    def save_all(
        self,
        base_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        format: str = 'jpg',
        sep: str = ',',
        output_message: bool = True,
        quality: int = 95,
    ) -> None:
        """
        Save the annotated image, morphology CSV, and color CSV in one call.

        Derives default output paths from :attr:`image_path` when
        ``base_name`` or ``output_dir`` are not provided. CSV files are only
        written when the corresponding result table is non-empty.

        Parameters
        ----------
        base_name : str or None, optional
            Stem used for all output filenames. If ``None``, derived from
            :attr:`image_path`. Default is ``None``.
        output_dir : str or None, optional
            Directory where all files are saved. If ``None``, derived from
            :attr:`image_path`. Default is ``None``.
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
            :attr:`image_path` is not set.
        RuntimeError
            If any file cannot be saved due to an unexpected error.
        """
        try:
            # Determine base name
            if base_name is None:
                if not self.image_path:
                    raise ValueError("Cannot determine base name: no original image available")
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

            # Determine output directory
            if output_dir is None:
                if not self.image_path:
                    raise ValueError("Cannot determine directory: no original image available")
                output_dir = os.path.dirname(self.image_path)
            
            # Ensure output directory exists (once)
            abs_output_dir = os.path.abspath(os.path.expanduser(output_dir))
            if abs_output_dir not in self._dir_cache:
                if not os.path.exists(abs_output_dir):
                    os.makedirs(abs_output_dir, exist_ok=True)
                self._dir_cache[abs_output_dir] = True
            
            # Build complete paths
            img_path = os.path.join(abs_output_dir, f"{base_name}_annotated.{format.lower()}")
            morph_csv_path = os.path.join(abs_output_dir, f"{base_name}_morphology_results.csv")
            color_csv_path = os.path.join(abs_output_dir, f"{base_name}_color_results.csv")
            
            # Save annotated image
            bgr_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
            
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(img_path, bgr_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                cv2.imwrite(img_path, bgr_image, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(img_path, bgr_image)
            
            if output_message:
                print(f"Image saved at: {img_path}")
            
            # Save morphology csv
            if isinstance(self.morphology_results, pd.DataFrame):
                morph_df = self.morphology_results
            else:
                morph_df = pd.DataFrame(self.morphology_results) if self.morphology_results else pd.DataFrame()

            if not morph_df.empty: 
                morph_df.to_csv(morph_csv_path, sep=sep, index=False, encoding='utf-8')
                if output_message:
                    print(f"Morphology CSV saved at: {morph_csv_path}")
            
            # Save color csv
            if isinstance(self.color_results, pd.DataFrame):
                color_df = self.color_results
            else:
                color_df = pd.DataFrame(self.color_results) if self.color_results else pd.DataFrame()
            
            if not color_df.empty:
                color_df.to_csv(color_csv_path, sep=sep, index=False, encoding='utf-8')
                if output_message:
                    print(f"Color CSV saved at: {color_csv_path}")
 
            
        except Exception as e:
            raise RuntimeError(f"Error in save_all: {str(e)}")
        
    def save_csv(
        self,
        path: Optional[str] = None,
        sep: str = ',',
        output_message: bool = True,
        data: str = 'auto',
        base_name: Optional[str] = None,
    ) -> None:
        """
        Save morphology results, color results, or both to CSV files.

        Resolves output paths from ``path``, ``base_name``, and
        :attr:`image_path`. Supports three path modes:

        - ``path=None`` — saves next to :attr:`image_path` using
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
            ``None``, derived from :attr:`image_path`, or ``'results'`` if
            :attr:`image_path` is not set. Default is ``None``.

        Raises
        ------
        ValueError
            If the requested result table is empty, ``path`` is ``None`` and
            :attr:`image_path` is not set, ``path`` has a non-CSV extension,
            or ``data`` is not one of the supported modes.
        """

        def to_df(obj) -> pd.DataFrame:
            """Robust conversion to DataFrame."""
            if obj is None:
                return pd.DataFrame()
            if isinstance(obj, pd.DataFrame):
                return obj
            # list[dict], dict, list, etc.
            try:
                return pd.DataFrame(obj)
            except Exception:
                return pd.DataFrame()

        morph_df = to_df(getattr(self, "morphology_results", None))
        color_df = to_df(getattr(self, "color_results", None))

        # Resolve default base_name
        if base_name is None:
            if self.image_path:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            else:
                base_name = "results"

        if path is None:
            if not self.image_path:
                raise ValueError("No path provided and no original image reference available")
            out_dir = os.path.dirname(self.image_path)
            base_path = os.path.join(out_dir, base_name)

        else:
            expanded = os.path.abspath(os.path.expanduser(path))

            # If user passed a directory, save inside it
            if os.path.isdir(expanded) or expanded.endswith(os.sep):
                out_dir = expanded.rstrip(os.sep)
                # ensure directory exists
                os.makedirs(out_dir, exist_ok=True)
                base_path = os.path.join(out_dir, base_name)

            else:
                # Treat as a file path (must end with .csv)
                full_path = self._ensure_dir_exists(expanded)
                stem, ext = os.path.splitext(full_path)
                if ext and ext.lower() != ".csv":
                    raise ValueError("save_csv path must end with .csv (or be a directory)")
                base_path = stem  # remove .csv so we can add suffixes consistently

        def save_df(df: pd.DataFrame, out_path: str, label: str) -> bool:
            """Save df if not empty. Returns True if saved."""
            if df is None or df.empty:
                return False
            if not out_path.lower().endswith(".csv"):
                out_path += ".csv"
            out_path = self._ensure_dir_exists(out_path)
            df.to_csv(out_path, sep=sep, index=False, encoding="utf-8", na_rep="NaN")
            if output_message:
                print(f"{label} CSV saved at: {out_path}")
            return True

        mode = (data or "auto").strip().lower()

        if mode == "morphology":
            if morph_df.empty:
                raise ValueError("No morphology results available to save")
            save_df(morph_df, f"{base_path}_morphology_results.csv", "Morphology")

        elif mode == "color":
            if color_df.empty:
                raise ValueError("No color results available to save")
            save_df(color_df, f"{base_path}_color_results.csv", "Color")

        elif mode == "both":
            saved_any = False
            saved_any |= save_df(morph_df, f"{base_path}_morphology_results.csv", "Morphology")
            saved_any |= save_df(color_df, f"{base_path}_color_results.csv", "Color")
            if not saved_any:
                raise ValueError("No morphology or color results available to save")

        elif mode == "auto":
            saved_any = False
            
            if not morph_df.empty:
                saved_any |= save_df(morph_df, f"{base_path}_morphology_results.csv", "Morphology")
            
            if not color_df.empty:
                saved_any |= save_df(color_df, f"{base_path}_color_results.csv", "Color")
            
            
            if not saved_any:
                raise ValueError("No morphology or color results available to save")

        else:
            raise ValueError("data must be one of: 'auto', 'morphology', 'color', 'both'")
