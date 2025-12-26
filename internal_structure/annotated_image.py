import cv2
import os
import pandas as pd
from typing import Optional
import numpy as np
from .. import valid_extensions

class AnnotatedImage:
    """
    Handles annotated images and results management.
    Stores analysis results and provides saving functionality.

    """
    
    def __init__(self, cv2_image: np.ndarray, results: list = None, image_path: Optional[str] = None):
        # Store both BGR (for cv2) and RGB (for display) to avoid reconversion
        self.bgr_image = cv2_image  # Original BGR format
        self.rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  
        self.results = results if results else []   
        self.table = self.results                  
        self.image_path = image_path
        self._dir_cache = {}  # Cache for directory checks

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
            
            # OPTIMIZED: Use cv2.imwrite directly (much faster than matplotlib)
            if format.lower() in ['jpg', 'jpeg']:
                # JPEG with quality setting
                cv2.imwrite(full_path, self.bgr_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                # PNG with compression
                cv2.imwrite(full_path, self.bgr_image, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                # Other formats - default
                cv2.imwrite(full_path, self.bgr_image)
            
            if output_message:
                print(f"Image saved at: {full_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error saving image: {str(e)}")

    def save_csv(self, path: Optional[str] = None, sep: str = ',', 
                 output_message: bool = True):
        """
        Save CSV in the same directory as the original image.
        
        Args:
            path (str, optional): Output path. If None, generated automatically.
            sep (str): CSV separator.
            output_message (bool): Whether to show confirmation message.
        """
        if not self.table:
            raise ValueError("No results data available to save")
        
        try:
            if path is None:
                if not self.image_path:
                    raise ValueError("No path provided and no original image reference available")
                
                original_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                path = os.path.join(original_dir, f"{base_name}_results.csv")
            
            full_path = self._ensure_dir_exists(path)
            
            df = pd.DataFrame(self.table)
            df.to_csv(full_path, sep=sep, index=False, encoding='utf-8')
            
            if output_message:
                print(f"CSV saved at: {full_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error saving CSV: {str(e)}")

    def save_all(self, base_name: Optional[str] = None, output_dir: Optional[str] = None, 
                 format: str = 'jpg', dpi: int = 75, sep: str = ',', 
                 output_message: bool = True, quality: int = 95):
        """
        Save both files (image and CSV) using the base name.
        OPTIMIZED: Inline logic to avoid redundant path operations.
        
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
            csv_path = os.path.join(abs_output_dir, f"{base_name}_results.csv")
            
            # OPTIMIZED: Inline save logic to avoid redundant operations
            # Save image with cv2
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(img_path, self.bgr_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                cv2.imwrite(img_path, self.bgr_image, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(img_path, self.bgr_image)
            
            if output_message:
                print(f"Image saved at: {img_path}")
            
            # Save CSV
            if self.table:
                df = pd.DataFrame(self.table)
                df.to_csv(csv_path, sep=sep, index=False, encoding='utf-8')
                
                if output_message:
                    print(f"CSV saved at: {csv_path}")
            else:
                raise ValueError("No results data available to save")
            
        except Exception as e:
            raise RuntimeError(f"Error in save_all: {str(e)}")