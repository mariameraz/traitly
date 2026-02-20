import streamlit as st
from traitly.fruit_phenotyping import FruitAnalyzer
import tempfile
import os

st.set_page_config(page_title="Fruit Analyzer", page_icon="🍊")

st.title("Fruit Analyzer (traitly)")
st.write(
    "Upload a cross-section fruit image and the app will run the full pipeline:\n"
    "**read → measure → create mask → find fruits → analyze → save results**"
)

# --- Upload image ---
uploaded_file = st.file_uploader(
    "Upload a fruit image:",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

# --- Main parameters ---
st.subheader("Main settings")

diameter_cm = st.number_input(
    "Real fruit diameter (cm)",
    min_value=0.1,
    value=1.7,
    step=0.1,
)

use_gpu = st.checkbox("Use GPU (if available)", value=True)
detect_label = st.checkbox("Detect label in the image", value=False)

# --- Advanced options ---
with st.expander("Advanced mask options"):
    roi_expansion = st.number_input(
        "ROI expansion",
        min_value=0,
        value=10,
        step=1,
    )
    locules_filled = st.checkbox(
        "Fill locules (internal cavities)",
        value=True,
    )
    canny_min = st.number_input(
        "Canny min",
        min_value=0,
        value=50,
        step=1,
    )
    canny_max = st.number_input(
        "Canny max",
        min_value=1,
        value=300,
        step=1,
    )
    n_blur = st.number_input(
        "Number of blurs",
        min_value=0,
        value=1,
        step=1,
    )

st.markdown("---")

run_button = st.button("Run analysis")

if run_button:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        # Save uploaded image temporarily
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                with st.spinner("Running FruitAnalyzer..."):
                    analyzer = FruitAnalyzer(img_path)

                    analyzer.read_image(plot=False)

                    analyzer.setup_measurements(
                        diameter_cm=diameter_cm,
                        gpu=use_gpu,
                        detect_label=detect_label,
                    )

                    analyzer.create_mask(
                        plot=False,
                        roi_expansion=roi_expansion,
                        locules_filled=locules_filled,
                        canny_min=canny_min,
                        canny_max=canny_max,
                        n_blur=n_blur,
                    )

                    analyzer.find_fruits()
                    analyzer.analyze_image()

                    # Save CSV + result images
                    analyzer.results.save_all()

                st.success("Done! Results saved with `save_all()`.")
                st.info(
                    "CSV and result images were saved to the output folder defined by traitly."
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
