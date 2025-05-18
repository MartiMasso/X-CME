import streamlit as st
import matplotlib as mpl
import imageio_ffmpeg

from A_data import (
    load_data,
    doy_2_datetime,
    display_data_info,
    display_file_details,
    identify_coordinate_system,
    handle_file_upload,
    resample_data,
)
from B_plotting import plot_data
from E_M1_Radial import fit_M1_radial
from E_M2_AngularRadial import fit_M2_AngularRadial
from E_M3_ExponAngular import fit_M3_ExponAngular
from E_Comparison import models_Comparison

# ─── Main setup for animation ─────────────────────────────────────────────────
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ─── Global CSS to tighten sidebar spacing ────────────────────────────────────
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] .stRadio {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── App header ───────────────────────────────────────────────────────────────
st.title("X-CME")
st.subheader("Flux Rope Fitting and CME Propagation Simulation - NASA")
st.markdown("**Author:** Martí Massó Moreno")

# ─── Sidebar: File upload options ─────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='margin-top:0.2rem; margin-bottom:0.2rem;'><strong>Load Options</strong></div>",
    unsafe_allow_html=True,
)
upload_option = st.sidebar.radio("", ("Upload a file", "Download from cloud"))
data, file_name = handle_file_upload(upload_option)

# ─── Sidebar: Display toggles ────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='margin-top:0.2rem; margin-bottom:0.2rem;'><strong>Display Options</strong></div>",
    unsafe_allow_html=True,
)
show_plots     = st.sidebar.checkbox("Compute detailed plots", value=True)
show_animation = st.sidebar.checkbox("Compute CME animation", value=True)

# ─── Sidebar: Model selection ────────────────────────────────────────────────
def get_fitting_model():
    st.sidebar.markdown(
        "<div style='margin-top:0.2rem; margin-bottom:0.2rem;'><strong>Choose Fitting Model</strong></div>",
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "",
        (
            "Model 1: EC - Nieves-Chinchilla - Radial Model",
            "Model 2: EC - Navas-Nieves-Masso - Radial and Poloidal Model",
        ),
    )

# ─── Sidebar: Iteration settings ─────────────────────────────────────────────
def get_iteration_params():
    max_n = 20
    powers_of_5 = [n**5 for n in range(3, max_n + 1)]
    N_iter_power = st.sidebar.select_slider(
        "Iterations for fitting (n⁵)",
        options=powers_of_5,
        value=powers_of_5[1],
        format_func=lambda x: f"{x:,}",
        help="Choose total iterations as fifth powers (e.g. 3⁵=243, 4⁵=1024)."
    )
    n_frames = st.sidebar.slider(
        "Frames in simulation",
        min_value=4,
        max_value=1000,
        value=20,
        help="Number of frames for CME propagation animation"
    )
    return int(round(N_iter_power ** (1/5))), n_frames

# ─── Process data ────────────────────────────────────────────────────────────
if data is not None:
    st.write("DataFrame shape before resampling:", data.shape)
    data = resample_data(data)
    st.write("DataFrame shape after resampling:", data.shape)

    coordinate_system = identify_coordinate_system(data)
    full_name, file_date, file_duration, file_year, mission = display_file_details(
        file_name, coordinate_system
    )
    display_data_info(data)

    # Data visualization
    (
        initial_date,
        final_date,
        filtered_data,
        total_days,
        hours,
        minutes,
        start_index,
        end_index,
    ) = plot_data(data, file_year)

    # Retrieve sidebar inputs
    fitting_model = get_fitting_model()
    N_iter, n_frames = get_iteration_params()

    # Execute fitting (moved to sidebar)
    if st.sidebar.button("Execute Fitting"):
        if fitting_model.startswith("Model 1"):
            st.write("Executing Fitting using Model 1...")
            (
                best_combination,
                B_components_fit,
                trajectory_vectors,
                viz_3d_vars_opt,
                viz_2d_local_vars_opt,
                viz_2d_rotated_vars_opt,
            ) = fit_M1_radial(
                show_plots,
                show_animation,
                data,
                start_index,
                end_index,
                initial_date,
                final_date,
                mission,
                N_iter,
                n_frames,
            )

        elif fitting_model.startswith("Model 2"):
            st.write("Executing Fitting using Model 2...")
            (
                best_combination,
                B_components_fit,
                trajectory_vectors,
                viz_3d_vars_opt,
                viz_2d_local_vars_opt,
                viz_2d_rotated_vars_opt,
            ) = fit_M2_AngularRadial(
                show_plots,
                show_animation,
                data,
                start_index,
                end_index,
                initial_date,
                final_date,
                mission,
                N_iter,
                n_frames,
            )
