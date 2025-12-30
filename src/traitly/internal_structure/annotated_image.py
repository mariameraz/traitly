import cv2
import os
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime
import time
import psutil

class AnnotatedImage:
    """
    Handles annotated images and results management.
    Stores analysis results and provides saving functionality.
    """
    
    def __init__(self, cv2_image: np.ndarray, results: list = None, 
                 image_path: Optional[str] = None, processing_metadata: Optional[Dict[str, Any]] = None):
        # Store both BGR (for cv2) and RGB (for display) to avoid reconversion
        self.bgr_image = cv2_image  # Original BGR format
        self.rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  
        self.results = results if results else []   
        self.table = self.results                  
        self.image_path = image_path
        self._dir_cache = {}  # Cache for directory checks
        
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

    def save_reports(self, output_dir: Optional[str] = None, 
                     output_message: bool = True):
        """
        Save error report and session report (identical format to analyze_folder).
        Handles both single image and batch processing scenarios.
        
        Args:
            output_dir (str, optional): Output directory. 
                If None, uses original image directory.
            output_message (bool): Whether to show confirmation messages.
        """
        try:
            # Determine output directory
            if output_dir is None:
                if not self.image_path:
                    raise ValueError("Cannot determine directory: no original image available")
                output_dir = os.path.dirname(self.image_path)
            
            # Ensure output directory exists
            abs_output_dir = os.path.abspath(os.path.expanduser(output_dir))
            if abs_output_dir not in self._dir_cache:
                if not os.path.exists(abs_output_dir):
                    os.makedirs(abs_output_dir, exist_ok=True)
                self._dir_cache[abs_output_dir] = True
            
            # Get metadata
            metadata = self.processing_metadata
            
            # Check if this is batch processing or single image
            is_batch = metadata.get('total_images', 1) > 1
            
            if is_batch:
                # Batch processing - error report is handled externally
                # Only save session report here
                self._save_batch_session_report(abs_output_dir, metadata, output_message)
            else:
                # Single image processing
                filename = os.path.basename(self.image_path) if self.image_path else 'unknown'
                
                # 1. ERROR REPORT (CSV format)
                error_dict = {
                    'filename': filename,
                    'status': 'Successfully processed' if self.results else 'No valid fruits detected'
                }
                
                error_csv_path = os.path.join(abs_output_dir, "error_report.csv")
                pd.DataFrame([error_dict]).to_csv(error_csv_path, index=False)
                
                if output_message:
                    print(f"Error report saved at: {error_csv_path}")
                
                # 2. SESSION REPORT (TXT format)
                self._save_single_session_report(abs_output_dir, metadata, output_message)
            
        except Exception as e:
            raise RuntimeError(f"Error saving reports: {str(e)}")
    
    def _save_single_session_report(self, output_dir: str, metadata: dict, output_message: bool):
        """Save session report for single image processing."""
        start_datetime = metadata.get('start_datetime', datetime.now())
        processing_time = metadata.get('processing_time', 0.0)
        ram_used = metadata.get('ram_used_gb', 0.0)
        n_fruits = len(self.results)
        config = metadata.get('config', {})
        
        separator = "══════════════════════════════════════════════"
        session_report_text = '\n'.join([
            separator,
            "SESSION DETAILS:",
            separator,
            f"date: {start_datetime.strftime('%Y-%m-%d')}",
            f"time: {start_datetime.strftime('%H:%M:%S')}",
            f"folder_path: {os.path.dirname(self.image_path) if self.image_path else 'N/A'}",
            f"total_images_detected: 1",
            f"processed_images: 1",
            f"total_fruits_processed: {n_fruits}",
            f"skipped_no_contours: 0",
            f"skipped_errors: 0",
            f"processing_time_minutes: {round(processing_time / 60, 2)}",
            f"avg_time_per_image_sec: {round(processing_time, 2)}",
            f"ram_used_gb: {round(ram_used, 2)}",
            "",
            separator,
            "ARGUMENTS:",
            separator,
            f"n_cores: 1",
            f"contour_mode: {config.get('contour_mode', 'raw')}",
            f"stamp: {config.get('stamp', False)}",
            f"min_circularity: {config.get('min_circularity', 0.3)}",
            f"min_locule_area: {config.get('min_locule_area', 300)}",
            f"merge_locules: {config.get('merge_locules', False)}",
            f"confidence_threshold: {config.get('confidence_threshold', 0.6)}",
            f"diameter_cm: {config.get('diameter_cm', 2.5)}",
            f"detect_label: {config.get('detect_label', True)}",
            f"use_ellipse_fruit: {config.get('use_ellipse', False)}"
        ])
        
        session_txt_path = os.path.join(output_dir, "session_report.txt")
        with open(session_txt_path, 'w', encoding='utf-8') as f:
            f.write(session_report_text + '\n')
        
        if output_message:
            print(f"Session report saved at: {session_txt_path}")
    
    def _save_batch_session_report(self, output_dir: str, metadata: dict, output_message: bool):
        """Save session report for batch processing."""
        start_datetime = metadata.get('start_datetime', datetime.now())
        path_input = self.image_path if os.path.isdir(self.image_path) else os.path.dirname(self.image_path)
        
        total_imgs = metadata.get('total_images', 0)
        processed_count = metadata.get('processed_images', 0)
        total_fruits = metadata.get('total_fruits', 0)
        skipped_no_contours = metadata.get('skipped_no_contours', 0)
        skipped_errors = metadata.get('skipped_errors', 0)
        processing_time = metadata.get('processing_time', 0.0)
        ram_used = metadata.get('ram_used_gb', 0.0)
        
        elapsed_min = processing_time / 60
        seg_per_img = processing_time / total_imgs if total_imgs > 0 else 0
        
        config = metadata.get('config', {})
        n_cores = config.get('n_cores', 1)
        
        separator = "══════════════════════════════════════════════"
        session_report_text = '\n'.join([
            separator,
            "SESSION DETAILS:",
            separator,
            f"date: {start_datetime.strftime('%Y-%m-%d')}",
            f"time: {start_datetime.strftime('%H:%M:%S')}",
            f"folder_path: {path_input}",
            f"total_images_detected: {total_imgs}",
            f"processed_images: {processed_count}",
            f"total_fruits_processed: {total_fruits}",
            f"skipped_no_contours: {skipped_no_contours}",
            f"skipped_errors: {skipped_errors}",
            f"processing_time_minutes: {round(elapsed_min, 2)}",
            f"avg_time_per_image_sec: {round(seg_per_img, 2)}",
            f"ram_used_gb: {round(ram_used, 2)}",
            "",
            separator,
            "ARGUMENTS:",
            separator,
            f"n_cores: {n_cores}",
            f"contour_mode: {config.get('contour_mode', 'raw')}",
            f"stamp: {config.get('stamp', False)}",
            f"min_circularity: {config.get('min_circularity', 0.3)}",
            f"min_locule_area: {config.get('min_locule_area', 300)}",
            f"merge_locules: {config.get('merge_locules', False)}",
            f"confidence_threshold: {config.get('confidence_threshold', 0.6)}",
            f"diameter_cm: {config.get('diameter_cm', 2.5)}",
            f"detect_label: {config.get('detect_label', True)}",
            f"use_ellipse_fruit: {config.get('use_ellipse_fruit', False)}"
        ])
        
        session_txt_path = os.path.join(output_dir, "session_report.txt")
        with open(session_txt_path, 'w', encoding='utf-8') as f:
            f.write(session_report_text + '\n')
        
        if output_message:
            print(f"Session report saved at: {session_txt_path}")

    def save_all(self, base_name: Optional[str] = None, output_dir: Optional[str] = None, 
                 format: str = 'jpg', dpi: int = 75, sep: str = ',', 
                 output_message: bool = True, quality: int = 95,
                 include_reports: bool = True):
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
            
            # Save reports if requested
            if include_reports:
                self.save_reports(output_dir=abs_output_dir, output_message=output_message)
            
        except Exception as e:
            raise RuntimeError(f"Error in save_all: {str(e)}")