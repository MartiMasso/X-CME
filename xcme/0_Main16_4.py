# 0_Main16_4.py

# ----------------------------------------------------------------
# Global patch to disable LaTeX in matplotlib (must come first)
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = False
mpl.rcParams['axes.formatter.limits'] = [-7, 7]
import matplotlib.texmanager as _tm
_tm.TeXManager.get_text_width_height_descent = lambda self, tex, fontsize: (0, 0, 0)
# ----------------------------------------------------------------

import streamlit as st
from A_data import load_data, doy_2_datetime, display_data_info, display_file_details, identify_coordinate_system, handle_file_upload, resample_data
from B_plotting import plot_data
from C_pdf_generator import generate_pdf_report
from E_M1_Radial import fit_M1_radial
from E_M2_AngularRadial import fit_M2_AngularRadial
from E_M3_CC_Radial import fit_M3_CC_radial
from E_Comparison import models_Comparison

st.title("Flux Rope Fitting")

upload_option = st.sidebar.radio("Load options:", ("Upload a file", "Download from cloud"))
data, file_name = handle_file_upload(upload_option)

if data is not None:
    st.write("DataFrame shape before resampling:", data.shape)
    data = resample_data(data)
    st.write("DataFrame shape after resampling:", data.shape)

    coordinate_system = identify_coordinate_system(data)
    full_name, file_date, file_duration, file_year, distance, lon_ecliptic = display_file_details(file_name, coordinate_system)
    display_data_info(data)

    initial_date, final_date, filtered_data, total_days, hours, minutes, start_index, end_index = plot_data(data, file_year)

    if st.sidebar.button("Generate PDF Report"):
        generate_pdf_report(data, file_name, file_date, file_duration, full_name, initial_date, final_date, filtered_data, coordinate_system)

    st.subheader("Select Fitting Model")
    fitting_model = st.radio("Choose the fitting model:", ("Model 1: EC - Only radial dependency", "Model 2: EC - Radial and Angular", "Model 3: CC - Radial Circular Cylindrical (simpler than M1)"))

    powers_of_5 = [n**5 for n in range(3, 21)]
    N_iter = st.select_slider("Number of iterations for fitting (n⁵)", options=powers_of_5, value=powers_of_5[1], format_func=lambda x: f"{x:,}", help="Choose iteration counts as powers of 5. Recommended: 10⁵ (100,000).")
    st.write(f"Selected iterations: {N_iter:,}")

    n_frames = st.slider("Number of frames in the simulation", min_value=4, max_value=1000, value=20, help="Frames for CME propagation")

    N_base = int(round(N_iter ** (1/5)))

    if st.button("Execute Fitting"):
        if fitting_model == "Model 1: EC - Only radial dependency":
            st.write("Running Model 1...")
            results = fit_M1_radial(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic, N_base, n_frames)
        elif fitting_model == "Model 2: EC - Radial and Angular":
            st.write("Running Model 2...")
            results = fit_M2_AngularRadial(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic, N_base, n_frames)
        elif fitting_model == "Model 3: CC - Radial Circular Cylindrical (simpler than M1)":
            st.write("Running Model 3...")
            results = fit_M3_CC_radial(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic, N_base, n_frames)
        # Uncomment to compare all models:
        # elif fitting_model == "Comparison of all models":
        #     st.write("Comparing all models...")
        #     results = models_Comparison(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic)

        # Process and display results here, e.g.:
        # best_combo, B_fit, traj, viz3d, viz2d_loc, viz2d_rot = results
        # st.write("Best combination:", best_combo)
