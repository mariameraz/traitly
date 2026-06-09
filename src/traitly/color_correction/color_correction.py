# traitly/color_correction/color_correction.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from typing import Tuple, Optional, Dict
import copy
import time
from datetime import datetime
import logging

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

from traitly.utils.batch import (
    _setup_batch,
    _print_batch_header,
    _run_color_batch_loop,
    _save_color_batch_results,
    _config_from_json
)

from traitly.utils.manage_params import _get_params, _clean_params

from .correction_parameters import ColorCorrectionParameters

logger = logging.getLogger(__name__)

def _process_color_worker(img_path: str, config: Dict, output_path: str, delta_e: bool = False):
    t0 = time.time()
    try:
        worker = ColorCorrection(img_path)
        worker.load_image(plot=False)
        err = worker.process_single_file(config=config)

        delta_df = None

        if err is None:
            worker.save_img(output_path=output_path, verbose=False)
            if delta_e:
                try:
                    worker.calculate_delta_e_stats(verbose=False)
                    delta_df = pd.DataFrame(
                        worker._delta_e_stats,
                        columns=["Patch", "Color", "DeltaE_Before",
                                 "DeltaE_After", "DeltaE_Improvement"],
                    )
                    for col in ["DeltaE_Before", "DeltaE_After", "DeltaE_Improvement"]:
                        delta_df[col] = delta_df[col].astype(float)
                    delta_df.insert(0, "Image_name", os.path.basename(img_path))
                except Exception:
                    logger.warning(
                            "Cannot calculate delta E for %s",
                            os.path.basename(img_path),
                            exc_info=True,
                                        )
                    delta_df = None

        return err, os.path.basename(img_path), time.time() - t0, delta_df

    except Exception as e:
        err = {"filename": os.path.basename(img_path), "status": f"Error: {str(e)}"}
        return err, os.path.basename(img_path), time.time() - t0, None


## Color correction class ##
class ColorCorrection:

    def __init__(self, path: str) -> None:
        # Get absolute path
        self.input_path = os.path.abspath(path)
        ## Verify path exists
        _validate_path_exists(self.input_path)

        # Determine if the path is to an image or a folder
        self._is_directory = os.path.isdir(path)

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

        logger.info("Processing %s", os.path.basename(self.input_path))

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
        logger.debug("Adjusting PLSR models (degree=%s, num_components=%s)", degree, num_components)
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
        logger.info("Applying color correction...")
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

        logger.info("Correction finished")
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

    def process_single_file(self, config=None):
        config = config or {}
        error_dict = None

        try:
            self.detect_color_checker(plot=False, verbose=False)
            if self._chart is None:
                raise RuntimeError("No color checker detected")
            self.apply_color_correction(
                plot=False, verbose=False,
                **_clean_params(_get_params(config, "apply_color_correction_params"))
            )

        except Exception as e:
            error_dict = {"filename": os.path.basename(self.input_path), "status": str(e)}

        return error_dict

    def analyze_folder(
        self,
        delta_e: bool = True,
        degree: Optional[int] = None,
        num_components: Optional[int] = None,
        max_iterations: Optional[int] = None,
        scaler=None,
        num_cores: int = 1,
        verbose: bool = True,
        json_path: Optional[str] = None,
        config: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ):
        folder_path, output_path, num_cores, num_cores_message, img_paths = _setup_batch(
            is_directory=self._is_directory,
            input_path=self.input_path,
            output_path=output_path,
            num_cores=num_cores,
        )

        # Configurate params
        config = copy.deepcopy(config) if config else {}
        _config_from_json(json_path, config)

        def _apply(section, mapping):
            overrides = {k: v for k, v in mapping.items() if v is not None}
            if overrides:
                config.setdefault(section, {})
                config[section].update(overrides)

        _apply("apply_color_correction_params", dict(
            degree=degree,
            num_components=num_components,
            max_iterations=max_iterations,
            scaler=scaler,
        ))

        if config.get("apply_color_correction_params"):
            setattr(self._parameters, "apply_color_correction_params", config["apply_color_correction_params"])

        session_start = datetime.now()

        _print_batch_header(
            folder_path=folder_path,
            img_paths=img_paths,
            num_cores=num_cores,
            num_cores_message=num_cores_message,
            verbose=verbose,
            json_path=json_path,
        )


        errors, per_image_times, all_delta = _run_color_batch_loop(
            img_paths=img_paths,
            worker_fn=_process_color_worker,
            num_cores=num_cores,
            config=config,
            output_path=output_path,
            verbose=verbose,
            delta_e=delta_e,
        )

        delta_before_mean = delta_after_mean = None

        if delta_e and all_delta:
            df_delta_all = pd.concat(all_delta, ignore_index=True)
            delta_csv = os.path.join(output_path, "delta_e_results.csv")
            df_delta_all.to_csv(delta_csv, index=False)

            delta_before_mean = df_delta_all["DeltaE_Before"].mean()
            delta_after_mean = df_delta_all["DeltaE_After"].mean()

        _save_color_batch_results(
            errors=errors,
            output_path=output_path,
            folder_path=folder_path,
            img_paths=img_paths,
            num_cores=num_cores,
            json_path=json_path,
            session_start=session_start,
            parameters=self._parameters,
            param_sections={"APPLY_COLOR_CORRECTION": "apply_color_correction_params"},
            verbose=verbose,
            delta_before_mean=delta_before_mean,
            delta_after_mean=delta_after_mean,
        )
