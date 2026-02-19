import cv2
import os
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime
import time
import psutil

class ResultsImage:
    """
    Handles annotated images and results management.
    Stores analysis results and provides saving functionality.
    """
    
    def __init__(self, 
                 bgr_img: np.ndarray, 
                 morphology_results: list = None, 
                 color_results: list = None, 
                 image_path: Optional[str] = None, 
                 processing_metadata: Optional[Dict[str, Any]] = None,
                 ):
        # Store both BGR (for cv2) and RGB (for display) to avoid reconversion
        self.annotated_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)    # Original BGR format
        #self.rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  
        self.morphology_results = morphology_results if morphology_results else []   
        self.table = self.morphology_results                  
        self.image_path = image_path
        self._dir_cache = {}  # Cache for directory checks
        self.color_results = color_results if color_results else []
        
        # Store metadata for reports
        self.processing_metadata = processing_metadata or {}

    def _ensure_dir_exists(self, path: str) -> str:
        """
        Ensure the directory exists and return the absolute path.
        (Uses caching to avoid repeated filesystem check)
        
        Args:
            path (str): File path to check
            
        Returns:
            str: Absolute path with ensured directory existence
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

    def save_img(self, path: Optional[str] = None, format: Optional[str] = None, 
                 dpi: int = 75, output_message: bool = True, quality: int = 95, **kwargs):
        """
        Save the image in the same directory as the original image.
        
        Args:
            path (str, optional): Output path. If None, generated automatically.
            format (str, optional): Image format. Defaults to extension inference.
            dpi (int): Resolution for raster formats (used only for format info).
            output_message (bool): Whether to show confirmation message.
            quality (int): JPEG quality (0-100). Default is 95.
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
    
    
    def save_all(self, base_name: Optional[str] = None, output_dir: Optional[str] = None, 
                 format: str = 'jpg', dpi: int = 75, sep: str = ',', 
                 output_message: bool = True, quality: int = 95):
        """
        Save all files (image, CSV, and reports) using the base name.
        
        Args:
            base_name (str, optional): Base name for files. 
                If None, uses original image name.
            output_dir (str, optional): Output directory. 
                If None, uses original image directory.
            format (str): Image format.
            dpi (int): Image resolution (for reference only).
            sep (str): CSV separator.
            output_message (bool): Whether to show confirmation messages.
            quality (int): JPEG quality (0-100). Default is 95.
            include_reports (bool): Whether to save error and session reports. Default is True.
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
            
            # ===== SAVE IMAGE (convertir RGB a BGR para cv2.imwrite) =====
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
            
            # ===== SAVE MORPHOLOGY CSV =====
            if isinstance(self.morphology_results, pd.DataFrame):
                morph_df = self.morphology_results
            else:
                morph_df = pd.DataFrame(self.morphology_results) if self.morphology_results else pd.DataFrame()

            if not morph_df.empty: 
                morph_df.to_csv(morph_csv_path, sep=sep, index=False, encoding='utf-8')
                if output_message:
                    print(f"Morphology CSV saved at: {morph_csv_path}")
            
            # ===== SAVE COLOR CSV =====
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
        sep: str = ",",
        output_message: bool = True,
        data: str = "auto",  # "auto" | "morphology" | "color" | "both"
        base_name: Optional[str] = None,
    ):
        """
        Save results to CSV.

        Parameters
        ----------
        path:
            - If None: saves next to the original image, using base_name.
            - If provided:
                * If endswith ".csv": treated as a *base file* (stem) for saving one or both CSVs.
                * If is a directory: saves inside that directory using base_name.
        data:
            - "morphology": saves morphology_results only
            - "color": saves color_results only
            - "both": saves both (two files)
            - "auto": saves morphology if available; else color if available
        base_name:
            Base name for files when path is None or a directory.
        """
        import os
        import pandas as pd

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
