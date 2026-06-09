# traitly/utils/batch.py
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import cv2
from typing import Optional, Tuple, List, Callable, Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pandas as pd
from tqdm import tqdm

# ============================================================================
# INTERNAL
# ============================================================================
from traitly import __version__
from traitly.utils.environment import get_package_versions
from traitly.utils.validation import (
    _validate_path_exists,
    _validate_num_cores,
    _valid_images_in_folder,
)

def _setup_batch(
    is_directory: bool,
    input_path: str,
    output_path: Optional[str],
    num_cores: int,
) -> Tuple[str, str, int, Optional[str], List[str]]:

    if not is_directory:
        raise ValueError(
            "analyze_folder() requires a directory path, not a single file."
        )

    folder_path = input_path
    _validate_path_exists(folder_path, makedir = False)

    num_cores, num_cores_message = _validate_num_cores(num_cores=num_cores)

    if output_path is None:
        results_path = os.path.join(folder_path, "Results")
        output_path = _validate_path_exists(results_path, makedir=True, base_name="")
    else:
        output_path = _validate_path_exists(output_path, makedir=True, base_name="")

    img_paths = _valid_images_in_folder(folder_path)

    return folder_path, output_path, num_cores, num_cores_message, img_paths

def _print_batch_header(
    folder_path: str,
    img_paths: List[str],
    num_cores: int,
    num_cores_message: Optional[str],
    verbose: bool,
    json_path: Optional[str] = None,
    extra_lines: Optional[List[str]] = None,
):
    if not verbose:
        return
    print("=" * 60)
    print(" Traitly running ⋆✧｡٩(ˊᗜˋ)و✧*｡")
    print("=" * 60)
    print(f"    > Input folder: {folder_path}")
    print(f"    > Image(s) detected: {len(img_paths)}")
    print(num_cores_message if num_cores_message else f"    > num_cores: {num_cores}"
    )
    if json_path is not None:
        print(f"    > Parameters loaded from: {os.path.abspath(json_path)}\n")
    if extra_lines:
        for line in extra_lines:
            if line is not None:
                print(line)

def _run_fruit_batch_loop(
    img_paths: List[str],
    worker_fn: Callable,
    parallel_worker_fn: Callable,
    num_cores: int,
    config: Dict,
    analyze_morphology: bool,
    analyze_color: bool,
    output_path: str,
    verbose: bool,
) -> Tuple[List, List, List, int, List]:
    all_morphology, all_color, errors = [], [], []
    total_fruits = 0
    per_image_times = []

    if num_cores == 1:
        for img_path in tqdm(img_paths, desc="Processing images", unit="img", disable=not verbose):
            err, fname, elapsed = worker_fn(img_path, config, output_path)
            per_image_times.append({"filename": fname, "time_s": round(elapsed, 2), "status": "error" if err else "ok", "fruits": n})
            if err:
                errors.append(err)
            else:
                if df_m is not None: all_morphology.append(df_m)
                if df_c is not None: all_color.append(df_c)
                total_fruits += n
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {
                executor.submit(parallel_worker_fn, img_path, config, analyze_morphology, analyze_color): img_path
                for img_path in img_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="img", disable=not verbose):
                result = future.result()
                df_m, df_c, err, n, ann_img, fname = result[:6]
                elapsed = result[6] if len(result) > 6 else 0.0
                per_image_times.append({"filename": fname, "time_s": round(elapsed, 2), "status": "error" if err else "ok", "fruits": n})
                if err:
                    errors.append(err)
                else:
                    if df_m is not None: all_morphology.append(df_m)
                    if df_c is not None: all_color.append(df_c)
                    total_fruits += n

    return all_morphology, all_color, errors, total_fruits, per_image_times


def _run_color_batch_loop(
    img_paths: List[str],
    worker_fn: Callable,
    parallel_worker_fn: Callable,
    num_cores: int,
    config: Dict,
    output_path: str,
    verbose: bool,
    delta_e: bool = False,
) -> Tuple[List, List, List]:

    errors = []
    per_image_times = []
    all_delta = []

    if num_cores == 1:
        for img_path in tqdm(img_paths, desc="Processing images", unit="img", disable=not verbose):
            err, fname, elapsed, delta_df = worker_fn(img_path, config, output_path, delta_e)
            per_image_times.append({
                "filename": fname,
                "time_s": round(elapsed, 2),
                "status": "error" if err else "ok",
            })
            if err:
                errors.append(err)
            elif delta_df is not None:
                all_delta.append(delta_df)
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {
                executor.submit(parallel_worker_fn, img_path, config, output_path, delta_e): img_path
                for img_path in img_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="img", disable=not verbose):
                err, fname, elapsed, delta_df = future.result()
                per_image_times.append({
                    "filename": fname,
                    "time_s": round(elapsed, 2),
                    "status": "error" if err else "ok",
                })
                if err:
                    errors.append(err)
                elif delta_df is not None:
                    all_delta.append(delta_df)

    return errors, per_image_times, all_delta

def _build_session_lines(
    session_start: datetime,
    folder_path: str,
    output_path: str,
    img_paths: List[str],
    errors: List[Dict],
    num_cores: int,
    json_path: Optional[str],
    extra_lines: Optional[List[str]] = None,
) -> Tuple[List[str], str, str]:
    session_end = datetime.now()
    total_time = (session_end - session_start).total_seconds()
    avg_time = total_time / len(img_paths) if img_paths else 0
    time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time / 60:.1f}min"
    json_report = os.path.abspath(json_path) if json_path else "No JSON file provided"

    lines = [
        "=" * 70, "SESSION REPORT", "=" * 70,
        f"traitly              : v{__version__}",
        f"run date             : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"image folder         : {folder_path}",
        f"results folder       : {output_path}",
        f"images found         : {len(img_paths)}",
        f"images ok            : {len(img_paths) - len(errors)}",
        f"images failed        : {len(errors)}",
        f"JSON path            : {json_report}",
        f"num_cores            : {num_cores}",
        f"total time           : {time_str}",
        f"avg per image        : {avg_time:.1f}s",
    ]

    if extra_lines:
        lines += extra_lines

    return lines, time_str, avg_time


def _build_error_report(
    errors: List[Dict],
    img_paths: List[str],
    output_path: str,
    session_start: datetime,
    folder_path: str,
) -> Optional[str]:
    if not errors:
        return None

    col1_w = max(len("IMAGE"), max(len(e["filename"]) for e in errors)) + 2
    col2_w = max(len("ERROR"), max(len(e["status"]) for e in errors)) + 2
    sep = f"+{'-' * col1_w}+{'-' * col2_w}+"
    header = f"| {'IMAGE':<{col1_w - 2}} | {'ERROR':<{col2_w - 2}} |"

    error_lines = [
        "=" * 70, "ERROR REPORT", "=" * 70,
        f"run date   : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"folder     : {folder_path}",
        f"failed     : {len(errors)}/{len(img_paths)} images",
        "", sep, header, sep,
    ] + [f"| {e['filename']:<{col1_w - 2}} | {e['status']:<{col2_w - 2}} |" for e in errors] + [sep]

    error_txt = os.path.join(output_path, "error_report.txt")
    with open(error_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(error_lines))


def _save_fruit_batch_results(
    all_morphology: List[pd.DataFrame],
    all_color: List[pd.DataFrame],
    errors: List[Dict],
    output_path: str,
    folder_path: str,
    img_paths: List[str],
    total_fruits: int,
    num_cores: int,
    analyze_morphology: bool,
    analyze_color: bool,
    json_path: Optional[str],
    session_start: datetime,
    parameters,
    param_sections: Dict,
    verbose: bool,
) -> None:

    # 1. Merge and save CSVs
    morph_csv = color_csv = None

    df_morph_all = pd.concat(all_morphology, ignore_index=True) if all_morphology else None
    df_color_all = pd.concat(all_color, ignore_index=True) if all_color else None

    if df_morph_all is not None:
        morph_csv = os.path.join(output_path, "morphology_results.csv")
        df_morph_all.to_csv(morph_csv, index=False)

    if df_color_all is not None:
        color_csv = os.path.join(output_path, "color_results.csv")
        df_color_all.to_csv(color_csv, index=False)

    # 2. Session report
    extra_lines = [
        "",
        f"total fruits         : {total_fruits}",
        f"analyze_morphology   : {analyze_morphology}",
        f"analyze_color        : {analyze_color}",
    ]

    session_lines, time_str, avg_time = _build_session_lines(
        session_start = session_start,
        folder_path = folder_path,
        output_path = output_path,
        img_paths = img_paths,
        errors = errors,
        num_cores = num_cores,
        json_path = json_path,
        extra_lines = extra_lines,
    )

    def _filter_params(p: Dict) -> Dict:
        return {k: v for k, v in p.items() if "plot" not in k.lower() and "color" not in k.lower()}

    session_lines += ["", "=" * 70, "ANALYSIS PARAMETERS", "=" * 70]
    for title, attr in param_sections.items():
        raw = getattr(parameters, attr, {}) or {}
        filtered = _filter_params(raw)
        if filtered:
            session_lines.append(f"\n{title}:")
            for k, v in filtered.items():
                session_lines.append(f"   - {k}: {v}")

    session_lines += [
        "", "=" * 70, "DEPENDENCIES", "=" * 70,
    ] + [f"   - {pkg:<30} {ver}" for pkg, ver in get_package_versions().items()]

    session_txt = os.path.join(output_path, "session_report.txt")
    with open(session_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(session_lines))

    # 3. Error report
    error_txt = _build_error_report(
        errors=errors,
        img_paths=img_paths,
        output_path=output_path,
        session_start=session_start,
        folder_path=folder_path,
    )

    # 4. Verbose
    if verbose:
        total_processed = len(img_paths) - len(errors)
        if len(errors) == len(img_paths):
            print("\n( ദ്ദി ༎ຶ‿༎ຶ ) Task failed successfully " + "=" * 37)
            print("    > Image(s) processed:")
            print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            print(f"    > For more details, check error_report.txt saved in: {output_path}")
        else:
            print("\n( ദ്ദി ˙ᗜ˙ ) Finished " + "=" * 47)
            print("    > Image(s) processed:")
            print(f"        - Successfully: {total_processed}/{len(img_paths)} img(s)")
            if errors:
                print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            print(f"        - Total fruits: {total_fruits}")
            print(f"        - Total time: {time_str}  (avg {avg_time:.1f}s/img)")
            print("    > Files saved:")
            print(f"        - {total_processed} annotated image(s)")
            if morph_csv:
                print(f"        - {os.path.basename(morph_csv)}")
            if color_csv:
                print(f"        - {os.path.basename(color_csv)}")
            print(f"        - {os.path.basename(session_txt)}")
            if error_txt:
                print(f"        - {os.path.basename(error_txt)}")
            print(f"        - Results folder: {output_path}")


def _save_color_batch_results(
    errors: List[Dict],
    output_path: str,
    folder_path: str,
    img_paths: List[str],
    num_cores: int,
    json_path: Optional[str],
    session_start: datetime,
    parameters,
    param_sections: Dict,
    verbose: bool,
    delta_before_mean: Optional[float] = None,
    delta_after_mean: Optional[float] = None,
) -> None:

    # 1. Session report
    session_lines, time_str, avg_time = _build_session_lines(
        session_start = session_start,
        folder_path = folder_path,
        output_path = output_path,
        img_paths = img_paths,
        errors = errors,
        num_cores = num_cores,
        json_path = json_path,
    )

    session_lines += ["", "=" * 70, "ANALYSIS PARAMETERS", "=" * 70]
    for title, attr in param_sections.items():
        raw = getattr(parameters, attr, {}) or {}
        if raw:
            session_lines.append(f"\n{title}:")
            for k, v in raw.items():
                session_lines.append(f"   - {k}: {v}")

    session_lines += [
        "", "=" * 70, "DEPENDENCIES", "=" * 70,
    ] + [f"   - {pkg:<30} {ver}" for pkg, ver in get_package_versions().items()]

    session_txt = os.path.join(output_path, "session_report.txt")
    with open(session_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(session_lines))

    # 2. Error report
    error_txt = _build_error_report(
        errors = errors,
        img_paths = img_paths,
        output_path = output_path,
        session_start = session_start,
        folder_path = folder_path,
    )

    # 3. verbose
    if verbose:
        total_processed = len(img_paths) - len(errors)
        if len(errors) == len(img_paths):
            print("\n( ദ്ദി ༎ຶ‿༎ຶ ) Task failed successfully " + "=" * 37)
            print("    > Image(s) processed:")
            print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            print(f"    > For more details, check error_report.txt saved in: {output_path}")
        else:
            print("\n( ദ്ദി ˙ᗜ˙ ) Finished " + "=" * 47)
            print("    > Image(s) processed:")
            print(f"        - Successfully: {total_processed}/{len(img_paths)} img(s)")
            if errors:
                print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            if delta_before_mean is not None:
                print(f"        - Mean ΔE before correction: {delta_before_mean:.2f}")
                print(f"        - Mean ΔE after correction:  {delta_after_mean:.2f}")
            print(f"        - Total time: {time_str}  (avg {avg_time:.1f}s/img)")
            if error_txt:
                print(f"        - {os.path.basename(error_txt)} created")
            print(f"        - Results folder: {output_path}")
