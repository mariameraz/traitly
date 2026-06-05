# traitly/color_correction/color_correction.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from typing import Tuple, Optional, Dict

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PolynomialFeatures
)

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.utils.session_report import _save_parameters

from traitly.utils.validation import (
    _validate_path_exists,
    _validate_color_image,
    _validate_img_suffix,
    _validate_num_cores,
)

from traitly.utils.save_results import _save_df, _save_img, _format_output_path
from traitly.utils.basic_functions import load_img, detect_img_name

from traitly.color_correction.color_analysis import (
    _detect_color_checker,
    _get_lab_patches,
    _fit_plsr_models,
    _img_bgr_to_lab,
    _apply_color_correction,
    _delta_e_stats,

)

from .correction_parameters import ColorCorrectionParameters

## Color correction class ##
class ColorCorrection:

    def __init__(self, path: str) -> None:
        # Get absolute path
        self.input_path = os.path.abspath(path)
        ## Verify path exists
        _validate_path_exists(self.input_path)

        # Determine if the path is to an image or a folder
        self._is_directory = os.path.isdir(os.path.dirname(path))

        # load_image
        self.original_img = None

        # detect_color_checker
        self._checker_coords = None
        self._chart = None
        self._img_copy = None
        self._detected_lab = None

        # apply_color_correction
        self._models = None
        self._img_lab = None
        self.corrected_img = None

        # calculate_delta_e_stats
        self._delta_e_stats = None

        # save_csv
        self._img_name = None

        # save parameters
        self._parameters = ColorCorrectionParameters()
        self._is_metadata_saved = True


    def load_image(
        self,
        plot: bool = True,
        plot_size: Tuple[int, int] = (5, 5),
        show_axis: bool = False,
    ):
        # validate valid image format
        _validate_img_suffix(self.input_path)

        self.original_img = load_img(
            self.input_path,
            plot=plot,
            plot_size=plot_size,
            show_axis=show_axis,
        )

        # check image loaded successfully
        _validate_color_image(self.original_img)

        return None

    def detect_color_checker(
        self,
        plot: bool = False,
        plot_size: Tuple[int, int] = (5,5),
        verbose: bool = True,
    ) -> None:
        # 1. Detect color checker
        self._checker_coords, self._chart, self._img_copy = _detect_color_checker(
            self.original_img,
            verbose=verbose,
            plot = plot,
            plot_size = plot_size)

        # 2. Extract color patches
        if self._chart is not None:
            self._detected_lab = _get_lab_patches(self._chart)

        return None

    def apply_color_correction(
        self,
        degree: int = 3,
        num_components: int = 11,
        max_iterations: int = 1000,
        scaler = StandardScaler(),
        plot: bool = True,
        plot_size: Tuple = (8,5),
        verbose: bool = True
    ) -> None:

        # Save the parameters used

        self._parameters.apply_color_correction_params = {
            "degree": degree,
            "num_components": num_components,
            "max_iterations": max_iterations,
            "scaler": scaler,
        }

        # Fit a PLSR model per LAB channel
        self._models = _fit_plsr_models(
            self._detected_lab,
            scaler = StandardScaler(),
            degree = degree,
            num_components = num_components,
            max_iterations = max_iterations
        )

        # Convert image from BGR to LAB
        self._img_lab = _img_bgr_to_lab(self.original_img)

        if verbose:
            print("=" * 65, flush=True)
            print("Correcting color, this may take a few seconds... ⋆✧｡٩(ˊᗜˋ)و✧*｡", flush=True)

        # Apply color correction to LAB image and return corrected BGR image
        self.corrected_img = _apply_color_correction(
            self._img_lab,
            self._models
        )

        if plot:
            plt.figure(figsize = plot_size)
            plt.subplot(1,2,1)
            plt.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(cv2.cvtColor(self.corrected_img, cv2.COLOR_BGR2RGB))
            plt.title("Corrected Image")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        if verbose:
            print("Correction finished!")
            print("=" * 65, flush=True)

        return None

    def calculate_delta_e_stats(
        self,
        verbose: bool = False
    )-> None:
        self._include_delta_e_stats = True

        self._delta_e_stats = _delta_e_stats(
            self.corrected_img,
            detected_lab = self._detected_lab,
            verbose = verbose,
        )

        return None

    def save_csv(
        self,
        output_path: Optional[str] = None,
        base_name: Optional[str] = None,
        sep: str = ",",
        verbose: bool = True,
    ) -> None:

        _output_path, _base_name = _format_output_path(
                self.input_path,
                base_name=base_name,
                suffix="_delta_e_stats",
                output_path=output_path
            )

        # Convert np.ndarray into pd.DF before save it
        df = pd.DataFrame(
                self._delta_e_stats,
                columns=["Patch", "Color", "DeltaE_Before", "DeltaE_After", "DeltaE_Improvement"]
            )
        DF_AVAILABLE = _save_df(
            df,
            output_path=os.path.join(_output_path, _base_name),  # pass full path
            base_name=_base_name, # base_name arg being ignored, I will fix this soon
            sep=sep
        )

        if not DF_AVAILABLE and verbose:
            print("Results are None")

        return None

    def save_img(
        self,
        output_path: Optional[str] = None,
        base_name: Optional[str] = None,
        format: str = 'png',
        quality: int = 100,
        verbose: bool = True,
    ):

        _output_path, _base_name,=  _format_output_path(
            self.input_path,
            base_name = base_name,
            suffix = "_corrected",
            output_path = output_path
        )

        _save_img(
            img=self.corrected_img,
            path=self.input_path,
            output_path = _output_path,
            format = format,
            verbose = verbose,
            quality = quality,
            base_name = _base_name,
        )

        return None

    def save_parameters(self, output_path=None):
        _save_parameters(self.input_path, self._parameters, output_path)

    def process_single_file(
        self,
        config: Optional[Dict] = None,
        json_path: Optional[str] = None,
    ):
        """
        Run the full color correction pipeline on the already loaded image
        """

        # 1. Load params from json file if passed
        params = _import_params(
            json_path = json_path,
            config = config
        )

        # 2. Create empty objs to save Results
        error_dict = None

        try:
            # Parameters for apply the color correction model
            self.apply_color_correction(
                verbose=False, **_clean_params(_get_params("apply_color_correction_params"))
            )
        except Exception as e:
            error_dict = {"filename": os.path.basename(self.input_path), "status": str(e)}
            raise RuntimeError(f"[apply_color_correction] {e}")

        return error_dict

    def analyze_folder(
        self,
        delta_e: bool = False,
    ):
        """
        Process all images in the folder passed to :class `ColorCorrection`.
        """

        # 1. Check if path is passing a directory
        if not self._is_directory:
            raise ValueError(
                "analyze_folder() requires a directory path. "
                "Pass a folder to ColorCorrection(), not a single file."
            )

        # 2. If so, check if it exists
        folder_path = self.input_path
        _validate_path_exists(folder_path, makedir = False)

        # 3. Check if num_cores is a valid number
        num_cores, num_cores_message = _validate_num_cores(num_cores=num_cores)

        # 4. If output_path is None, create a new results folder
        if output_path is None:
            output_path = _validate_path_exists(folder_path, makedir=True)
        else:
            output_path = _validate_path_exists(output_path, makedir=True)

        # 5. Obtain the paths for all the valid images in the input folder
        img_paths = _valid_images_in_folder(folder_path)
