# tests/test_analyze_folder.py

######################### 
# Third-party libraries #
#########################
from pathlib import Path
import pandas as pd
import pytest

####################
# Internal library #
####################
from traitly.fruit_phenotyping import FruitAnalyzer

################################
# Test analyze_folder function #
################################

def test_analyze_folder_creates_outputs():
    # Folder with 3 example images.
    # 2 images have QR codes, 1 image does not have a QR code
    # Results are going to be saved by default inside the sample_data/Results folder.
    # If the Results folder does not exist, it will be created.
    # The image: "img_test_4.jpg" doesn't contain 'valid' locules. 
    # So, the function will not stop and only it will report the error.

    sample_data = Path("tests/sample_data")

    analyzer = FruitAnalyzer(sample_data)
    analyzer.analyze_folder()  # Run the folder analysis - It will process all images in the folder

    results_dir = sample_data / "Results"

    # Checks:
    # 1) output directory was created
    assert results_dir.is_dir()

    # 2) error_report.csv exists
    error_report = results_dir / "error_report.csv"
    assert error_report.exists()

    # 3) all_results.csv exists
    all_results = results_dir / "all_results.csv"
    assert all_results.exists()

    # 4) Has at least one row of results
    df = pd.read_csv(all_results)
    assert len(df) > 0

###############################
# Test running a single image #
###############################

def test_analyze_folder_raises_if_not_directory():
    # Provide a single image path instead of a directory
    example_img = "tests/sample_data/img_test_3.jpg"

    # Initialize FruitAnalyzer with a single image
    analyzer = FruitAnalyzer(str(example_img))

    with pytest.raises(ValueError, match="single image"):
        analyzer.analyze_folder()

######################
# Test analyze_image #
######################

def test_analyze_image_creates_outputs():
    # Path to a single image
    # QR code is present in this image
    # Results are going to be saved in Image_Results folder. If the folder does not exist, it will be created.

    sample_data = "tests/sample_data/img_test_1.jpg"

    analyzer = FruitAnalyzer(sample_data)
    analyzer.read_image() # Load the image
    analyzer.setup_measurements() # Setup measurements
    analyzer.create_mask() # Create the mask
    analyzer.find_fruits() # Find fruits in the image
    analyzer.analyze_image(plot = False)  # Run the image analysis
    analyzer.results.save_all(output_dir = "tests/sample_data/Image_Results")
    

    # Checks:
    results_dir = Path("tests/sample_data/Image_Results")

    # 1) error_report.csv exists
    # error_report = results_dir / "error_report.csv"
    # assert error_report.exists()

    # 2) annotated image exists
    annotated_img = results_dir / "img_test_1_annotated.jpg"
    assert annotated_img.exists()

    # 3) results.csv exists
    results = results_dir / "img_test_1_results.csv"
    assert results.exists()

    # 4) reports exist
    error_report = results_dir / "error_report.csv"
    assert error_report.exists()

    session_report = results_dir / "session_report.txt"
    assert session_report.exists()

    # 4) Has at least one row of results
    df = pd.read_csv(results)
    assert len(df) > 0


######################################
# Test analyzing an image with no QR #
######################################

def test_analyze_image_with_no_qr():
    # Path to an image without a QR code
    # Results are going to be saved in Image_Results folder
    # "No label detected/included" message is expected in the image_name column of the results.csv

    sample_data = "tests/sample_data/img_test_5.jpg"

    analyzer = FruitAnalyzer(sample_data)
    analyzer.read_image() # Load the image
    analyzer.setup_measurements(detect_label = True) # Setup measurements
    analyzer.create_mask() # Create the mask
    analyzer.find_fruits() # Find fruits in the image
    analyzer.analyze_image(plot = False)  # Run the image analysis
    analyzer.results.save_all(output_dir = "tests/sample_data/Image_Results") 

    # Checks:
    results_dir = Path("tests/sample_data/Image_Results")

    # 1) error_report.csv exists
    # error_report = results_dir / "error_report.csv"
    # assert error_report.exists()

    # 2) annotated image exists
    annotated_img = results_dir / "img_test_5_annotated.jpg"
    assert annotated_img.exists()

    # 3) results.csv exists
    results = results_dir / "img_test_5_results.csv"
    assert results.exists()

    # 4) reports exist
    error_report = results_dir / "error_report.csv"
    assert error_report.exists()

    session_report = results_dir / "session_report.txt"
    assert session_report.exists()

    # 4) Has at least one row of results
    df = pd.read_csv(results)
    assert len(df) > 0

##########################
# Test image with stamps #
##########################

######################################
# Test analyzing an image with no QR #
######################################

def test_analyze_image_with_stamps():
    # Path to an image with stamps instead of slices, without a QR code, and without label detection
    # Results are going to be saved in Stamp_Results folder (if the folder does not exist, it will be created)
    # "No label detected/included" message is expected in the image_name column of the results.csv

    sample_data = "tests/sample_data/stamp/img_test_4.jpg"

    analyzer = FruitAnalyzer(sample_data)
    analyzer.read_image() # Load the image
    analyzer.setup_measurements(detect_label = False) # Setup measurements
    analyzer.create_mask(stamp = True) # Create the mask
    analyzer.find_fruits() # Find fruits in the image
    analyzer.analyze_image(plot = False)  # Run the image analysis
    analyzer.results.save_all(output_dir = "tests/sample_data/Stamp_Results") 

    # Checks:
    results_dir = Path("tests/sample_data/Stamp_Results")

    # 1) error_report.csv exists
    # error_report = results_dir / "error_report.csv"
    # assert error_report.exists()

    # 2) annotated image exists
    annotated_img = results_dir / "img_test_4_annotated.jpg"
    assert annotated_img.exists()

    # 3) results.csv exists
    results = results_dir / "img_test_4_results.csv"
    assert results.exists()

    # 4) reports exist
    error_report = results_dir / "error_report.csv"
    assert error_report.exists()

    session_report = results_dir / "session_report.txt"
    assert session_report.exists()

    # 4) Has at least one row of results
    df = pd.read_csv(results)
    assert len(df) > 0
