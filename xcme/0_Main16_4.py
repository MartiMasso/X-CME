import streamlit as st
import matplotlib as mpl
import imageio_ffmpeg
from A_data import load_data, doy_2_datetime, display_data_info, display_file_details, identify_coordinate_system, handle_file_upload, resample_data
from B_plotting import plot_data
# from C_pdf_generator import generate_pdf_report
from E_M1_Radial import fit_M1_radial
from E_M2_AngularRadial import fit_M2_AngularRadial
from E_M3_ExponAngular import fit_M3_ExponAngular
from E_Comparison import models_Comparison


# Main
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
st.title("Flux Rope Fitting")

# File Upload Management
upload_option = st.sidebar.radio("Load options:", ("Upload a file", "Download from cloud"))
data, file_name = handle_file_upload(upload_option)

# Display and Analyze Data
if data is not None:
    st.write("DataFrame shape before resampling:", data.shape)
    data = resample_data(data)
    st.write("DataFrame shape after resampling:", data.shape)
    coordinate_system = identify_coordinate_system(data)
    full_name, file_date, file_duration, file_year,  distance, lon_ecliptic = display_file_details(file_name, coordinate_system)
    display_data_info(data)

    # Plot the data
    initial_date, final_date, filtered_data, total_days, hours, minutes, start_index, end_index = plot_data(data, file_year)
    
    # Generate PDF report
    # if st.sidebar.button("Generate PDF Report"):
    #     generate_pdf_report(data, file_name, file_date, file_duration, full_name, initial_date, final_date, filtered_data, coordinate_system)

    # Model Selection for Fitting
    st.subheader("Select Fitting Model")
    fitting_model = st.radio("Choose the fitting model:", (
        "Model 1: EC - Only radial dependency",
        "Model 2: EC - Radial and Angular",
        "Model 3: EC - Exponential and Angular",
        "Comparison of all models"
    ))

    # Generate the list of n**5 values
    max_n = 20
    powers_of_5 = [n**5 for n in range(3, max_n + 1)]  # [1, 32, 243, 1024, 3125, 7776]

    # Instead of st.slider, use st.select_slider for non‑linear steps
    if fitting_model:
        N_iter = st.select_slider(
            "Number of iterations for fitting (n⁵)",
            options=powers_of_5,
            value=powers_of_5[1],  # default is 2**5 = 32
            format_func=lambda x: f"{x:,}",
            help=(
                "Choose the total iterations as fifth powers (1⁵, 2⁵, 3⁵, …). "
                "Recommended to use 10⁵ (100,000 iterations). Results under 16807 (7⁵) may be inaccurate. "
                "We iterate through five geometric variables, using the same iteration count for each."
                " Estimated time for 100.000 iterations: 5 minutes. "
                "Estimated time for 3.200.000 iterations: 2h 40min. "
            )
        )
        st.write(f"Selected iterations: {N_iter}")
        n_frames = st.slider(
            "Number of frames in the simulation",
            min_value=4,
            max_value=1000,
            value=20,
            help="Frames for the propagation of the CME in the simulation"
        )

        N_iter = int(N_iter ** (1/5))  # Convert back to the base value for further calculations

        # Execute Data Fitting based on Model Selection
        if st.button("Execute Fitting"):
            if fitting_model == "Model 1: EC - Only radial dependency":
                st.write("Executing Fitting using Model 1...")
                best_combination, B_components_fit, trajectory_vectors, viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt = fit_M1_radial(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic, N_iter, n_frames)
            
            elif fitting_model == "Model 2: EC - Radial and Angular":
                st.write("Executing Fitting using Model 2...")
                best_combination, B_components_fit, trajectory_vectors, viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt = fit_M2_AngularRadial(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic,  N_iter, n_frames)

            elif fitting_model == "Model 3: EC - Exponential and Angular":
                st.write("Executing Fitting using Model 3...")
                best_combination, B_components_fit, trajectory_vectors, viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt = fit_M3_ExponAngular(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic)

            elif fitting_model == "Comparison of all models":
                st.write("Executing Comparison of all models...")
                best_combination, B_components_fit, trajectory_vectors, viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt = models_Comparison(data, start_index, end_index, initial_date, final_date, distance, lon_ecliptic)
    