import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage.measure import EllipseModel
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.path import Path
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from matplotlib.patches import Arc
from sympy.printing.latex import LatexPrinter 
from A_data import identify_coordinate_system
from matplotlib import colors
import os

# Necessary libraries for propagation of the CME
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import matplotlib.animation as animation
from A_data import get_position_data
from tqdm import tqdm
import astropy.units as u
from sunpy.coordinates import HeliocentricEarthEcliptic, get_horizons_coord, get_body_heliographic_stonyhurst
from sunpy.time import parse_time

def set_axes_equal(ax):
    """Sets the plot axes to have the same scale."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# --- Elliptical coordinate transformation function ---
def elliptical_coords(x, y, xc, yc, a, b, theta):
    X = x - xc
    Y = y - yc
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    Xp = X * cos_t - Y * sin_t
    Yp = X * sin_t + Y * cos_t
    r = np.sqrt((Xp / a)**2 + (Yp / b)**2)
    alpha = np.arctan2(Yp / b, Xp / a)
    # alpha = np.mod(alpha, 2*np.pi)
    return r, alpha

# Function to compute the fitting (with cache)
st.cache_data.clear()
@st.cache_data
def fit_M1_radial(data, initial_point, final_point, initial_date, final_date,  distance, lon_ecliptic, N_iter, n_frames):
    coordinate_system = identify_coordinate_system(data)

    data_transformed = data.copy()

    # If in RTN, transform to GSE
    if coordinate_system == "RTN":
        # Transform from RTN to GSE
        data_transformed['Bx'] = -data['Br']  # Bx (GSE) = -Br (RTN)
        data_transformed['By'] = data['Bt']   # By (GSE) = Bt (RTN)
        data_transformed['Bz'] = data['Bn']   # Bz (GSE) = Bn (RTN)
        # The magnitude B remains unchanged, it's already in data['B']
    elif coordinate_system != "GSE":
        raise ValueError("Unknown coordinate system. Expected GSE or RTN.")

    # Extract the transformed data (now always in GSE)
    ddoy_data = data_transformed['ddoy'].values
    B_data   = data_transformed['B'].values
    Bx_data  = data_transformed['Bx'].values
    By_data  = data_transformed['By'].values
    Bz_data  = data_transformed['Bz'].values
    Vsw_data = data_transformed['Vsw'].values
    data_tuple = (ddoy_data, B_data, Bx_data, By_data, Bz_data)

    # Logic for Np: use it if present; otherwise default to 60
    if 'Np' in data_transformed.columns:
        Np_data = data_transformed['Np'].values
    else:
        # Create an array the same size as ddoy_data filled with 60s
        Np_data = 60 * np.ones_like(ddoy_data)

    
    initial_point = max(1, initial_point)
    Npt = final_point - initial_point
    Npt_resolution = 400

    B_exp = B_data[initial_point - 1:final_point - 1]
    Bx_exp = Bx_data[initial_point - 1:final_point - 1]
    By_exp = By_data[initial_point - 1:final_point - 1]
    Bz_exp = Bz_data[initial_point - 1:final_point - 1]
    ddoy_exp = ddoy_data[initial_point - 1:final_point - 1]
    Vsw_exp = Vsw_data[initial_point - 1:final_point - 1]
    Np_exp = Np_data[initial_point - 1:final_point - 1]

    # Relevant average and geometric properties
    proton_mass = 1.6726219e-27  # kg
    density_avg = np.nanmean(Np_exp) * proton_mass * 1e6  # kg/m^3
    Vsw_avg = np.nanmean(Vsw_exp)
    Vsw = Vsw_avg
    time_span_fr = (ddoy_exp[-1] - ddoy_exp[0]) * 24 * 3600
    length_L1 = np.abs(Vsw_avg * time_span_fr)
    print(f"Densidad másica promedio: {density_avg:.2e} kg/m³")
    print(f"Vsw promedio: {Vsw_avg:.2e} kg/m³")

    # ### Test L1 data ### 
    # days = 2
    # ts = days * 3600 * 24
    # Vsw = 430
    # length_L1 = ts*Vsw

    def project_to_plane(point, normal, d):
        dot_product = np.dot(normal, point)
        t = (d - dot_product) / np.dot(normal, normal)
        projected_point = point + t * normal
        return projected_point

    # # Fixed test parameters (as in your original code)
    # z0_range = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])  # (-1, 1)
    # angle_x_range = np.array([np.radians(70)])              # (0, 180)
    # angle_y_range = np.array([np.radians(0)])               # (0, 180)
    # angle_z_range = np.array([-np.radians(0)])              # (0, 180)
    # delta_range = np.array([0.7])                           # (0, 1)

    # Fixed parameters for LaTeX illustration
    # z0_range = np.array([0.21])  # (-1, 1)
    # angle_x_range = np.array([np.radians(76)])              # (0, 180)
    # angle_y_range = np.array([np.radians(-1)])             # (0, 180)
    # angle_z_range = np.array([-np.radians(-30.1)])             # (0, 180)
    # delta_range = np.array([0.7])                           # (0, 1)



    # Iteration parameters
    z0_range = np.linspace(-0.7, 0.7, N_iter)                            # Relative entrance altitude (-1, 1), but we consider the top and bottom problematic in reality, so we use (-0.8, 0.8)
    angle_x_range = np.linspace(0.4, np.pi - 0.4, N_iter)                # Interval (0, π)
    angle_y_range = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, N_iter)   # Interval (-π/2, π/2)
    angle_z_range = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, N_iter)   # Interval (-π/2, π/2)
    delta_range = np.linspace(0.5, 1.0, N_iter)                          # Ellipse distortion (we do not consider more extreme distortions, as would be < 0.4).


    total_iterations = len(z0_range) * len(angle_x_range) * len(angle_y_range) * len(angle_z_range) * len(delta_range)
    st.write("Total Iterations:", total_iterations)

    best_R2 = -np.inf
    best_combination = None
    B_components_fit = None
    trajectory_vectors = None
    viz_3d_vars_opt = None
    viz_2d_local_vars_opt = None
    viz_2d_rotated_vars_opt = None

    progress_bar = st.progress(0)
    progress_text = st.empty()
    current_iteration = 0


    # Create a directory to store plots
    output_dir = "/Users/martimasso/Desktop/NASA/FR-Fitting-NASA/PlotsSaved"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Counter for unique filenames (optional, can use parameters instead)
    iteration_counter = 0

    angle_min_x_deg = 10  # Minimum angle of the FR axis wrt the x-axis. 
    angle_min_x = np.deg2rad(angle_min_x_deg) 
    angle_min_z_deg = 10  # Minimum angle of the FR axis wrt the z-axis. 
    angle_min_z = np.deg2rad(angle_min_z_deg) 

    # ----------------------------------------------------------------------------------
    # Main Loop           
    # ----------------------------------------------------------------------------------
    for z0 in z0_range:
        for angle_x in angle_x_range:
            for angle_y in angle_y_range:
                if abs(np.sin(angle_y) * np.cos(angle_x)) <= np.cos(angle_min_x) and abs(np.cos(angle_x) * np.cos(angle_y)) <= np.cos(angle_min_z):
                    for angle_z in angle_z_range:
                        for delta in delta_range:
                            current_iteration += 1
                            progress_percent = int((current_iteration / total_iterations) * 100)
                            progress_bar.progress(progress_percent)
                            progress_text.text(f"Processing... {progress_percent}% completed")
                            iteration_counter += 1
                            # ----------------------------------------------------------------------------------
                            # 1. Geometry configuration
                            # ----------------------------------------------------------------------------------
                            N = 100
                            a = 1
                            b = delta * a

                            # 1.1 Rotation Matrices
                            # ----------------------------------------------------------------------------------
                            R_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
                            R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
                            R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
                            rotation_matrix = R_y @ R_x @ R_z
                            rotation_matrix_inv = rotation_matrix.T

                            # 1.2 Cylinder Coordinates Centered at Origin
                            # ----------------------------------------------------------------------------------
                            theta = np.linspace(0, 2 * np.pi, N)
                            z = np.linspace(-1.5 * a, 1.5 * a, N)
                            Theta, Z = np.meshgrid(theta, z)
                            X = a * np.cos(Theta)
                            Y = b * np.sin(Theta)

                            # 1.3 Rotate Cylinder
                            # ----------------------------------------------------------------------------------
                            X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
                            rotated_points = np.dot(rotation_matrix, np.vstack([X_flat, Y_flat, Z_flat]))
                            X_rot, Y_rot, Z_rot = rotated_points[0].reshape(N, N), rotated_points[1].reshape(N, N), rotated_points[2].reshape(N, N)

                            # ----------------------------------------------------------------------------------
                            # 2. Cylinder Intersection with Plane y = 0 and Spacecraft
                            # ----------------------------------------------------------------------------------
                            cut_indices = np.abs(Y_rot) < 0.05
                            X_cut = X_rot[cut_indices]
                            Z_cut = Z_rot[cut_indices]

                            # 2.1 Sort Intersection Points
                            # ----------------------------------------------------------------------------------
                            angles_filtered = np.arctan2(Z_cut, X_cut)
                            sorted_indices_filtered = np.argsort(angles_filtered)
                            X_final = X_cut[sorted_indices_filtered]
                            Z_final = Z_cut[sorted_indices_filtered]

                            X_reduced = X_final[::3]
                            Z_reduced = Z_final[::3]

                            # 2.2 Fit Ellipse to Intersection
                            # ----------------------------------------------------------------------------------
                            ellipse_model = EllipseModel()
                            ellipse_model.estimate(np.column_stack((X_reduced, Z_reduced)))
                            xc, zc, a_ellipse, b_ellipse, theta = ellipse_model.params

                            # 2.3 Generate Ellipse Points
                            # ----------------------------------------------------------------------------------
                            t = np.linspace(0, 2 * np.pi, Npt_resolution)
                            X_ellipse = a_ellipse * np.cos(t) * np.cos(theta) - b_ellipse * np.sin(t) * np.sin(theta)
                            Z_ellipse = a_ellipse * np.cos(t) * np.sin(theta) + b_ellipse * np.sin(t) * np.cos(theta)

                            # 2.4 Center the Cylinder and Ellipse at Origin
                            # ----------------------------------------------------------------------------------
                            X_rot_centered = X_rot - xc
                            Y_rot_centered = Y_rot  # Y remains unchanged for y=0 plane
                            Z_rot_centered = Z_rot - zc

                            # ----------------------------------------------------------------------------------
                            # 3. Spacecraft Trajectory (horizontal cut) in the Section Ellipse
                            # ----------------------------------------------------------------------------------
                            Z_max = np.max(Z_ellipse)
                            Z_min = np.min(Z_ellipse)
                            z_cut = z0 * Z_max

                            # 3.1 Solve the System
                            # ----------------------------------------------------------------------------------
                            A_quad = (np.cos(theta)**2) / a_ellipse**2 + (np.sin(theta)**2) / b_ellipse**2
                            B_quad = 2 * ((np.cos(theta) * np.sin(theta)) / a_ellipse**2 - (np.cos(theta) * np.sin(theta)) / b_ellipse**2) * z_cut
                            C_quad = ((np.sin(theta)**2) / a_ellipse**2 + (np.cos(theta)**2) / b_ellipse**2) * z_cut**2 - 1

                            discriminant = B_quad**2 - 4 * A_quad * C_quad

                            if discriminant >= 0:
                                x1 = (-B_quad + np.sqrt(discriminant)) / (2 * A_quad) #+ xc
                                x2 = (-B_quad - np.sqrt(discriminant)) / (2 * A_quad) #+ xc

                                X_intersections = np.array([x1, x2])
                                Z_intersections = np.full_like(X_intersections, z_cut)

                                # 3.2 Scale Coordinates
                                # ----------------------------------------------------------------------------------
                                # Calculate the actual distance between intersection points
                                L_actual = np.abs(x2 - x1)

                                # Determine the scaling factor
                                scale_factor = length_L1 / L_actual

                                # Apply the scaling factor to the ellipse
                                X_ellipse_scaled = xc + scale_factor * (X_ellipse) #- xc)
                                Z_ellipse_scaled = zc + scale_factor * (Z_ellipse)# - zc)

                                idx_max = np.argmax(Z_ellipse_scaled)
                                idx_min = np.argmin(Z_ellipse_scaled)
                                x_at_Z_max = X_ellipse_scaled[idx_max]
                                x_at_Z_min = X_ellipse_scaled[idx_min]

                                # Scale the intersection points
                                x1_scaled = xc + scale_factor * (x1)# - xc)
                                x2_scaled = xc + scale_factor * (x2)# - xc)

                                Z_max_scaled = zc + scale_factor * (Z_max )#- zc)
                                z_cut_scaled = zc + scale_factor * (z_cut)# - zc)

                                X_intersections_scaled = np.array([x1_scaled, x2_scaled])
                                Z_intersections_scaled = np.full_like(X_intersections_scaled, z_cut_scaled)

                                # Scale the cylinder (only once)
                                X_rot_scaled = X_rot_centered * scale_factor
                                Y_rot_scaled = Y_rot_centered * scale_factor
                                Z_rot_scaled = Z_rot_centered * scale_factor

                                # 3.3 Satellite Trajectory
                                # ----------------------------------------------------------------------------------
                                xs = np.linspace(x1_scaled, x2_scaled, num=100)  # X points along the cut
                                ys = np.zeros_like(xs)  # y=0 plane
                                zs = np.full_like(xs, z_cut_scaled)  # Constant height at z_cut_scaled

                                # 3.4 Ellipse Boundaries and Reference Lines
                                # ----------------------------------------------------------------------------------
                                x_min = np.min(X_ellipse_scaled)
                                x_max = np.max(X_ellipse_scaled)
                                z_min = np.min(Z_ellipse_scaled)
                                z_max = np.max(Z_ellipse_scaled)

                                # Adjust small margins so the ellipse doesn’t touch the edges
                                margin = 0.1 * (x_max - x_min)  # 10% margin of the range
                                x_limits = [x_min - margin, x_max + margin]
                                z_limits = [z_min - margin, z_max + margin]

                                # Find the highest point of the ellipse
                                max_z_index = np.argmax(Z_ellipse_scaled)
                                x_max_z = X_ellipse_scaled[max_z_index]  # X at the maximum Z point

                                # Calculate limits of the lines within the ellipse
                                indices_z_zero = np.where(np.abs(Z_ellipse_scaled - 0) < 1e-2)[0]
                                x_z_zero_limits = np.sort(X_ellipse_scaled[indices_z_zero]) if len(indices_z_zero) >= 2 else [x_min, x_max]
                                indices_x_max_z = np.where(np.abs(X_ellipse_scaled - x_max_z) < 1e-2)[0]
                                z_x_max_z_limits = np.sort(Z_ellipse_scaled[indices_x_max_z]) if len(indices_x_max_z) >= 2 else [z_min, z_max]

                                # Calculate the upper half of the vertical line
                                z_mid = (z_x_max_z_limits[0] + z_x_max_z_limits[1]) / 2
                                z_upper_limit = z_x_max_z_limits[1]
                                z_lower_limit_upper_half = z_mid
                                height_upper_half = z_upper_limit - z_lower_limit_upper_half
                                relative_position = z_cut_scaled - z_lower_limit_upper_half
                                percentage_in_upper_half = (relative_position / height_upper_half) * 100 if height_upper_half > 0 else 0

                                # st.write(f"Percentage of the trajectory height in the upper half of maximum z: {percentage_in_upper_half:.2f}%")

                            else:
                                print("\nNo real solutions found for the cut at Z =", z_cut)
                                exit()

                            # ----------------------------------------------------------------------------------
                            # 4. Projection onto the Transverse Plane
                            # ----------------------------------------------------------------------------------
                        
                            # 4.1 Compute Cylinder Axis and Plane Equation
                            # ----------------------------------------------------------------------------------
                            axis_cylinder = rotation_matrix @ np.array([0, 0, 1])

                            center_point = np.array([xc, 0, zc])
                            d = np.dot(axis_cylinder, center_point)
                            # print(f"Transverse plane constant: d = {d:.6f}")

                            axis_cylinder_norm = axis_cylinder / np.linalg.norm(axis_cylinder)

                            # ----------------------------------------------------------------------
                            # Projections to the plane: 
                            # ----------------------------------------------------------------------

                            def project_to_plane(point, normal, d):
                                dot_product = np.dot(normal, point)
                                t = (d - dot_product) / np.dot(normal, normal)
                                projected_point = point + t * normal
                                return projected_point
                            
                            # 4.2 Project the ellipse
                            # ----------------------------------------------------------------------
                            X_ellipse_3d = np.vstack([X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled]).T
                            projected_ellipse = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in X_ellipse_3d])
                            X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse = projected_ellipse[:, 0], projected_ellipse[:, 1], projected_ellipse[:, 2]

                            # 4.3 Project the intersection points
                            # ----------------------------------------------------------------------
                            intersection_points = np.array([[x1_scaled, 0, z_cut_scaled], [x2_scaled, 0, z_cut_scaled]])
                            projected_intersections = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in intersection_points])
                            X_proj_inter, Y_proj_inter, Z_proj_inter = projected_intersections[:, 0], projected_intersections[:, 1], projected_intersections[:, 2]

                            # 4.4 Project the cut trajectory
                            # ----------------------------------------------------------------------
                            n_points = len(Bx_exp)
                            X_traj = np.linspace(x1_scaled, x2_scaled, n_points)
                            Y_traj = np.zeros_like(X_traj)
                            Z_traj = np.full_like(X_traj, z_cut_scaled)
                            trajectory_points = np.vstack([X_traj, Y_traj, Z_traj]).T
                            projected_trajectory = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in trajectory_points])
                            X_proj_traj, Y_proj_traj, Z_proj_traj = projected_trajectory[:, 0], projected_trajectory[:, 1], projected_trajectory[:, 2]

                            # 4.5 Fit an ellipse to the centered projected ellipse points
                            # ----------------------------------------------------------------------
                            ellipse_model = EllipseModel()
                            ellipse_points = np.column_stack((X_proj_ellipse, Y_proj_ellipse))
                            ellipse_model.estimate(ellipse_points)
                            xc_proj, yc_proj, a_proj, b_proj, theta_proj = ellipse_model.params
                            xc_proj, yc_proj = 0, 0  # Already centered at the origin

                            # Calculate the slope of the projected trajectory
                            # ----------------------------------------------------------------------
                            slope_traj = (Y_proj_traj[-1] - Y_proj_traj[0]) / (X_proj_traj[-1] - X_proj_traj[0]) if (X_proj_traj[-1] - X_proj_traj[0]) != 0 else float('inf')
                            trajectory_angle = np.arctan2(Y_proj_traj[-1] - Y_proj_traj[0], X_proj_traj[-1] - X_proj_traj[0])


                            # ----------------------------------------------------------------------------------
                            # 5. Express in Local Cartesian Coordinates, well orientated
                            # ----------------------------------------------------------------------------------
                            local_coords = np.dot(rotation_matrix_inv, projected_trajectory.T).T
                            x_local = local_coords[:, 0]
                            y_local = local_coords[:, 1]
                            z_local = local_coords[:, 2]

                            # Intersection points in local coords
                            intersections_local = np.dot(rotation_matrix_inv, intersection_points.T).T
                            x_inter_local = intersections_local[:, 0]
                            y_inter_local = intersections_local[:, 1]

                            # ----------------------------------------------------------------------------------
                            # (5A) ELLIPSE CONTOUR IN LOCAL COORDS
                            # ----------------------------------------------------------------------------------
                            ellipse_3d = np.vstack([X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled]).T
                            ellipse_local = np.dot(rotation_matrix_inv, ellipse_3d.T).T
                            x_ellipse_local = ellipse_local[:, 0]
                            y_ellipse_local = ellipse_local[:, 1]

                            # ----------------------------------------------------------------------------------
                            # (5B) ROTATION to align the trajectory horizontally
                            # ----------------------------------------------------------------------------------
                            dx = x_local[-1] - x_local[0]
                            dy = y_local[-1] - y_local[0]
                            trajectory_angle = np.arctan2(dy, dx) + np.pi

                            rotation_2d = np.array([
                                [np.cos(-trajectory_angle), -np.sin(-trajectory_angle)],
                                [np.sin(-trajectory_angle),  np.cos(-trajectory_angle)]
                            ])

                            # Rotate the ELLIPSE in local coords
                            ellipse_local_2d = np.vstack([x_ellipse_local, y_ellipse_local]).T
                            ellipse_rotated = (rotation_2d @ ellipse_local_2d.T).T
                            x_ellipse_rotated = ellipse_rotated[:, 0]
                            y_ellipse_rotated = ellipse_rotated[:, 1]

                            # Rotate the TRAJECTORY
                            trajectory_local_2d = np.vstack([x_local, y_local]).T
                            trajectory_rotated = (rotation_2d @ trajectory_local_2d.T).T
                            x_traj_rotated = trajectory_rotated[:, 0]
                            y_traj_rotated = trajectory_rotated[:, 1]

                            x_traj = x_traj_rotated
                            y_traj = y_traj_rotated

                            # Rotate the INTERSECTION POINTS
                            intersections_rotated = (rotation_2d @ np.vstack([x_inter_local, y_inter_local])).T
                            x_inter_rotated = intersections_rotated[:, 0]
                            y_inter_rotated = intersections_rotated[:, 1]

                            # ----------------------------------------------------------------------------------
                            # (EXTRA) DEFINE CHORDS and ROTATE THEM
                            # ----------------------------------------------------------------------------------
                            a_local = np.max(np.abs(x_ellipse_local))  # horizontal semi-axis (major axis)
                            b_local = np.max(np.abs(y_ellipse_local))  # vertical semi-axis (minor axis)

                            # Horizontal chord in local coords: y=0
                            h_line_local = np.array([[-a_local, 0],
                                                    [ a_local, 0]])
                            # Vertical chord in local coords: x=0
                            v_line_local = np.array([[0, -b_local],
                                                    [0,  b_local]])

                            # Rotate them
                            h_line_rotated = (rotation_2d @ h_line_local.T).T
                            v_line_rotated = (rotation_2d @ v_line_local.T).T

                            # We will treat h_line_rotated[1] as the "positive side" of the chord
                            # i.e., the end that extends in the positive-X direction in the local coords
                            chord_xr, chord_yr = h_line_rotated[1]  # endpoint of horizontal chord in rotated coords

                            # Angle of the chord in the rotated frame
                            chord_angle_right = np.degrees(np.arctan2(chord_yr, chord_xr)) % 360

                            # ------------------------------------------------------------------------------
                            # EXAMPLE: SAVE INTERSECTION ANGLE 0 AND 1, THEN LINSPACE
                            # ------------------------------------------------------------------------------
                            # Suppose we want the angles from the LEFT subplot for Intersection 0 and Intersection 1:
                            angle_0 = np.degrees(np.arctan2(y_inter_local[0], x_inter_local[0])) % 360
                            angle_1 = np.degrees(np.arctan2(y_inter_local[1], x_inter_local[1])) % 360

                            # Ensure we go from smaller to larger
                            start_angle = min(angle_0, angle_1)
                            end_angle   = max(angle_0, angle_1)

                            angles_traj = np.linspace(start_angle, end_angle, 10)

                            # # ----------------------------------------------------------------------------------
                            # # 6. 3D Representation
                            # # ----------------------------------------------------------------------------------
                            # (7) TRAJECTORY VECTORS
                            # # ----------------------------------------------------------------------------------
                            xc, zc, theta = 0, 0, 0

                            # Define the elliptical_coords function (already provided)
                            def elliptical_coords(x, z, xc, zc, a, b, theta):
                                X = x - xc
                                Z = z - zc
                                cos_t = np.cos(-theta)
                                sin_t = np.sin(-theta)
                                Xp = X * cos_t - Z * sin_t
                                Zp = X * sin_t + Z * cos_t
                                r = np.sqrt((Xp / a)**2 + (Zp / b)**2)
                                alpha = np.arctan2(Zp / b, Xp / a)
                                alpha = np.mod(alpha, 2 * np.pi)
                                return r, alpha

                            # Compute elliptical coordinates for the trajectory
                            r_vals, phi_vals = elliptical_coords(x_local, y_local, xc, zc, a_local, b_local, theta)

                            # Compute elliptical coordinates for the intersection points
                            r_inter, phi_inter = elliptical_coords(x_inter_local, y_inter_local, xc, zc, a_local, b_local, theta)

                            # Use the intersection angles from elliptical coordinates for linspace
                            angle_0 = np.degrees(phi_inter[0])  # 80.06°
                            angle_1 = np.degrees(phi_inter[1])  # 178.97°

                            # Ensure we go from smaller to larger angle
                            start_angle = min(angle_0, angle_1)
                            end_angle = max(angle_0, angle_1)

                            # Generate 10 equally spaced angles in elliptical coordinate space
                            angles_traj = np.linspace(start_angle, end_angle, 10)
                            phi_vals_deg = np.degrees(phi_vals)



                            # ----------------------------------------------------------------------------------
                            # 8. Express the in-situ data in the Local Cartesian Coordinate System
                            # ----------------------------------------------------------------------------------
                            # Transform GSE data to Local Cartesian coordinates using the inverse rotation matrix
                            Bx_GSE_exp = Bx_exp[::-1]
                            By_GSE_exp = By_exp[::-1]
                            Bz_GSE_exp = Bz_exp[::-1]
                            B_GSE_exp_tot = np.sqrt(Bx_GSE_exp**2 + By_GSE_exp**2 + Bz_GSE_exp**2)

                            B_gse_exp = np.vstack([Bx_exp, By_exp, Bz_exp])
                            x_traj_GSE = x_traj[::-1]

                            # --- Transform exported GSE components back to local coordinates ---
                            B_local_exp = rotation_matrix @ np.vstack((Bx_GSE_exp, By_GSE_exp, Bz_GSE_exp))
                            Bx_Local_exp = -B_local_exp[0, :]
                            By_Local_exp = -B_local_exp[2, :]
                            Bz_Local_exp = B_local_exp[1, :]
                            B_Local_total_exp = np.sqrt(Bx_Local_exp**2 + By_Local_exp**2 + Bz_Local_exp**2)

                            # ----------------------------------------------------------------------------------
                            # 9. Express the in-situ data in the Local CYLINDRICAL Coordinate System
                            # ----------------------------------------------------------------------------------


                            a_ell = 1

                            # Compute metric components along the trajectory
                            grr_traj = a_ell**2 * (np.cos(phi_vals)**2 + delta**2 * np.sin(phi_vals)**2)
                            gyy_traj = np.ones_like(r_vals)
                            gphiphi_traj = a_ell**2 * r_vals**2 * (np.sin(phi_vals)**2 + delta**2 * np.cos(phi_vals)**2)
                            grphi_traj = a_ell**2 * r_vals * np.sin(phi_vals) * np.cos(phi_vals) * (delta**2 - 1)

                            # --- Define the transformation coefficients ---
                            cos_phi = np.cos(phi_vals)
                            sin_phi = np.sin(phi_vals)
                            coeff1 = a_ell * cos_phi          # a * cos(φ)
                            coeff2 = -a_ell * r_vals * sin_phi  # -a * r * sin(φ)
                            coeff3 = a_ell * delta * sin_phi    # a * δ * sin(φ)
                            coeff4 = a_ell * delta * r_vals * cos_phi  # a * δ * r * cos(φ)

                            # --- Transform to cylindrical coordinates by solving the system ---
                            Br_exp = np.zeros_like(r_vals)
                            Bphi_exp = np.zeros_like(r_vals)
                            determinants = []  # To track determinant stability

                            for i in range(len(r_vals)):
                                A_matrix = np.array([[coeff1[i], coeff2[i]], [coeff3[i], coeff4[i]]])
                                det = np.linalg.det(A_matrix)
                                determinants.append(det)  # Store determinant for diagnostics
                                b_vector = np.array([Bx_Local_exp[i], Bz_Local_exp[i]])
                                if np.abs(det) > 1e-10:  # Check for invertibility
                                    Br_Bphi = np.linalg.solve(A_matrix, b_vector)
                                    Br_exp[i] = Br_Bphi[0]  # B^r
                                    Bphi_exp[i] = Br_Bphi[1]  # B^φ
                                else:
                                    Br_exp[i] = 0
                                    Bphi_exp[i] = 0
                                    print(f"Warning: Singular matrix at index {i}, det = {det:.6e}")
                            By_exp_cyl = By_Local_exp 

                            # --- Compute the modulus in cylindrical coordinates for exported data ---
                            B_total_exp_cyl = np.sqrt(
                                grr_traj * Br_exp**2 +
                                gyy_traj * By_exp**2 +
                                gphiphi_traj * Bphi_exp**2 +
                                2 * grphi_traj * Br_exp * Bphi_exp
                            )
    
                            
                            # ----------------------------------------------------------------------------------
                            # 10. Fitting Magnetic Field Models to Noisy Cylindrical Data
                            # ----------------------------------------------------------------------------------
                            
                            # 10.1. Models for the fitting
                            #------------------------------------------------------------------------------------------------
                            def model_Br(r_alpha, a):
                                a = 0
                                r, alpha = r_alpha
                                return np.zeros_like(r)

                            def model_By(r_alpha, A, B):
                                r, alpha = r_alpha
                                r = r * a_local * 10**3
                                return A + B * r**2

                            def model_Bphi(r_alpha, C):
                                r, alpha = r_alpha
                                r = r * a_local * 10**3
                                return C * r

                            # Data preparation (keeping your original structure)
                            data_fit = np.vstack((r_vals, phi_vals)).T

                            # Initial guesses for the fitting
                            initial_guess_Br = [0.0]  # For Br (although it’s fixed at 0)
                            initial_guess_By = [1.0, 1.0]  # [A, B]
                            initial_guess_Bphi = [1.0]  # [C]

                            try:
                                # Curve fitting for each component
                                params_Br, _ = curve_fit(model_Br, data_fit.T, Br_exp, p0=initial_guess_Br)
                                a_Br = params_Br[0]  # Extract the scalar value
                                Br_fitted = model_Br(data_fit.T, a_Br)

                                params_By, _ = curve_fit(model_By, data_fit.T, By_exp_cyl, p0=initial_guess_By)  # Use By_exp_cyl
                                A_By, B_By = params_By
                                By_fitted = model_By(data_fit.T, A_By, B_By)

                                params_Bphi, _ = curve_fit(model_Bphi, data_fit.T, Bphi_exp, p0=initial_guess_Bphi)
                                C_Bphi = params_Bphi[0]
                                Bphi_fitted = model_Bphi(data_fit.T, C_Bphi)

                                # Calculation of R² for each component
                                ss_tot_Br = np.sum((Br_exp - np.mean(Br_exp))**2)
                                ss_res_Br = np.sum((Br_exp - Br_fitted)**2)
                                R2_Br = 1 - (ss_res_Br / ss_tot_Br) if ss_tot_Br != 0 else 0

                                ss_tot_By = np.sum((By_exp_cyl - np.mean(By_exp_cyl))**2)
                                ss_res_By = np.sum((By_exp_cyl - By_fitted)**2)
                                R2_By = 1 - (ss_res_By / ss_tot_By) if ss_tot_By != 0 else 0

                                ss_tot_Bphi = np.sum((Bphi_exp - np.mean(Bphi_exp))**2)
                                ss_res_Bphi = np.sum((Bphi_exp - Bphi_fitted)**2)
                                R2_Bphi = 1 - (ss_res_Bphi / ss_tot_Bphi) if ss_tot_Bphi != 0 else 0

                                R2_avg = (R2_Br + R2_By + R2_Bphi) / 3

                                # Fitted vectors
                                Br_vector = model_Br(data_fit.T, a_Br)
                                By_vector = model_By(data_fit.T, A_By, B_By)
                                Bphi_vector = model_Bphi(data_fit.T, C_Bphi)

                                B_vector = np.sqrt(
                                    grr_traj * Br_vector**2 +
                                    gyy_traj * By_vector**2 +
                                    gphiphi_traj * Bphi_vector**2 +
                                    2 * grphi_traj * Br_vector * Bphi_vector
                                )

                            except RuntimeError:
                                st.write("Error in curve fitting: an optimal solution could not be found")


                            # ----------------------------------------------------------------------------------
                            # 11. Go back to local and then GSE coordinate system (for the fitted values)
                            # ----------------------------------------------------------------------------------

                            # --- Transformation to local Cartesian coordinates ---
                            Bx_traj = Br_vector * a_ell * np.cos(phi_vals) - Bphi_vector * a_ell * r_vals * np.sin(phi_vals)
                            By_traj_cartesian = By_vector  # By is already aligned with the local y-axis
                            Bz_traj = Br_vector * a_ell * delta * np.sin(phi_vals) + Bphi_vector * delta * a_ell * r_vals * np.cos(phi_vals)
                            B_vector = np.sqrt(Bx_traj**2 + By_traj_cartesian**2 + Bz_traj**2)

                            # --- Transformation to GSE coordinates (requires rotation_matrix) ---
                            B_local = np.vstack((-Bx_traj, Bz_traj, -By_traj_cartesian))  # Shape: (3, N_points)
                            B_GSE = rotation_matrix_inv @ B_local  # Shape: (3, N_points)
                            Bx_GSE = B_GSE[0, :]
                            By_GSE = B_GSE[1, :]
                            Bz_GSE = B_GSE[2, :]
                            B_total_GSE = np.sqrt(Bx_GSE**2 + By_GSE**2 + Bz_GSE**2)

                            x_traj_GSE = x_traj[::-1]

                            # ----------------------------------------------------------------------------------
                            # 12. Compute quality factor for the original in-situ data
                            # ----------------------------------------------------------------------------------

                            # Slice original data to match the fitted segment
                            B_data_segment = B_data[initial_point - 1:final_point - 1]
                            Bx_data_segment = Bx_data[initial_point - 1:final_point - 1]
                            By_data_segment = By_data[initial_point - 1:final_point - 1]
                            Bz_data_segment = Bz_data[initial_point - 1:final_point - 1]

                            B_data_segment = B_data_segment[::-1]
                            Bx_data_segment = Bx_data_segment[::-1]
                            By_data_segment = By_data_segment[::-1]
                            Bz_data_segment = Bz_data_segment[::-1]

                            # Calculation of the coefficient of determination R² (goodness-of-fit)
                            # For the total magnetic field strength (B)
                            ss_tot_B = np.sum((B_data_segment - np.mean(B_data_segment))**2)
                            ss_res_B = np.sum((B_data_segment - B_vector)**2)
                            R2_B = 1 - (ss_res_B / ss_tot_B) if ss_tot_B != 0 else 0

                            # For the Bx component
                            ss_tot_Bx = np.sum((Bx_data_segment - np.mean(Bx_data_segment))**2)
                            ss_res_Bx = np.sum((Bx_data_segment - Bx_GSE)**2)
                            R2_Bx = 1 - (ss_res_Bx / ss_tot_Bx) if ss_tot_Bx != 0 else 0

                            # For the By component
                            ss_tot_By = np.sum((By_data_segment - np.mean(By_data_segment))**2)
                            ss_res_By = np.sum((By_data_segment - By_GSE)**2)
                            R2_By = 1 - (ss_res_By / ss_tot_By) if ss_tot_By != 0 else 0

                            # For the Bz component
                            ss_tot_Bz = np.sum((Bz_data_segment - np.mean(Bz_data_segment))**2)
                            ss_res_Bz = np.sum((Bz_data_segment - Bz_GSE)**2)
                            R2_Bz = 1 - (ss_res_Bz / ss_tot_Bz) if ss_tot_Bz != 0 else 0

                            # Calculation of the average R²
                            R2_avg = (R2_B + R2_Bx + R2_By + R2_Bz) / 4


                            # ----------------------------------------------------------------------------------

                            # Update best combination if current R2 is better
                            if R2_avg > best_R2:
                                best_R2 = R2_avg

                                B_vector = B_vector[::-1]
                                Bx_GSE   = Bx_GSE[::-1]
                                By_GSE   = By_GSE[::-1]
                                Bz_GSE   = Bz_GSE[::-1]
                                
                                # # Parameter text for annotations with updated order
                                # param_text = (
                                #     f"Iter: {iteration_counter}\n"
                                #     r"$\theta_x$: " + f"{angle_x:.2f} rad\n"
                                #     r"$\theta_y$: " + f"{angle_y:.2f} rad\n"
                                #     r"$\theta_z$: " + f"{angle_z:.2f} rad\n"
                                #     f"z0: {z0:.2f}\n"
                                #     r"$\delta$: " + f"{delta:.2f}"
                                # )

                                # # ----------------------------------------------------------------------------------
                                # # Plot Combined) Cylindrical Components (Left) and Original Data Fitting (Right)
                                # # ----------------------------------------------------------------------------------
                                # # Creamos figura más grande para que los subplots ocupen más espacio
                                # fig, axes = plt.subplots(
                                #     4, 2,
                                #     figsize=(24, 18),                  # de 20x15 a 24x18
                                #     gridspec_kw={'width_ratios': [1, 1]}
                                # )

                                # # ─── Columna izquierda: Componentes cilíndricas ────────────────────────────────────
                                # marker_size = 10
                                # # Br
                                # axes[0, 0].scatter(x_traj, Br_exp,      color='blue', label=r"$B_r^{exp}$", s=marker_size)
                                # axes[0, 0].plot(   x_traj, Br_vector,   color='cyan', linestyle='--', label=r"$B_r^{fit}$")
                                # axes[0, 0].set_title("Radial Component $B_r$")
                                # axes[0, 0].set_xlabel("a")
                                # axes[0, 0].set_ylabel(r"$B_r$ (nT)")
                                # axes[0, 0].legend()
                                # axes[0, 0].grid(True)
                                # # By
                                # axes[1, 0].scatter(x_traj, By_exp_cyl,      color='green', label=r"$B_y^{exp}$", s=marker_size)
                                # axes[1, 0].plot(   x_traj, By_vector,   color='lime',  linestyle='--', label=r"$B_y^{fit}$")
                                # axes[1, 0].set_title("Axial Component $B_y$")
                                # axes[1, 0].set_xlabel("a")
                                # axes[1, 0].set_ylabel(r"$B_y$ (nT)")
                                # axes[1, 0].legend()
                                # axes[1, 0].grid(True)
                                # # Bphi
                                # axes[2, 0].scatter(x_traj, Bphi_exp,      color='red',    label=r"$B_\phi^{exp}$", s=marker_size)
                                # axes[2, 0].plot(   x_traj, Bphi_vector,   color='orange', linestyle='--', label=r"$B_\phi^{fit}$")
                                # axes[2, 0].set_title("Azimuthal Component $B_\phi$")
                                # axes[2, 0].set_xlabel("a")
                                # axes[2, 0].set_ylabel(r"$B_\phi$ (nT)")
                                # axes[2, 0].legend()
                                # axes[2, 0].grid(True)
                                # # Eliminamos el cuarto eje vacío
                                # axes[3, 0].axis('off')

                                # # ─── Columna derecha: Datos originales vs. ajustados ───────────────────────────────
                                # adjusted_data = [B_vector, Bx_GSE, By_GSE, Bz_GSE]
                                # components    = ['B', 'Bx', 'By', 'Bz']
                                # data_compare  = [B_data, Bx_data, By_data, Bz_data]
                                # titles_compare = [
                                #     "Magnetic Field Intensity (B)",
                                #     "Magnetic Field Component Bx",
                                #     "Magnetic Field Component By",
                                #     "Magnetic Field Component Bz",
                                # ]
                                # start_segment = ddoy_data[initial_point - 1]
                                # end_segment   = ddoy_data[final_point - 2]

                                # for idx, (comp, orig, adj, title) in enumerate(zip(components, data_compare, adjusted_data, titles_compare)):
                                #     ax = axes[idx, 1]
                                #     ax.scatter(ddoy_data, orig, color='black', s=10, label=f'{comp} Original')
                                #     ax.plot(
                                #         ddoy_data[initial_point - 1: final_point - 1],
                                #         adj,
                                #         color='red', linestyle='--', linewidth=2, label=f'{comp} Fitted'
                                #     )
                                #     ax.axvline(x=start_segment, color='gray', linestyle='--', label='Start of Segment')
                                #     ax.axvline(x=end_segment,   color='gray', linestyle='--', label='End of Segment')
                                #     ax.set_title(title, fontsize=14, fontweight='bold')
                                #     ax.set_ylabel(f"{comp} (nT)", fontsize=12)
                                #     if idx == 3:
                                #         ax.set_xlabel("Day of the Year (ddoy)", fontsize=12)
                                #     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                                #     ax.minorticks_on()
                                #     ax.legend(fontsize=10)

                                # # ─── Anotaciones de parámetros ────────────────────────────────────────────────────
                                # fig.text(
                                #     0.02, 0.03,            # posición, ligeramente movida para no recortar
                                #     param_text,
                                #     fontsize=28,           # tamaño de fuente duplicado
                                #     bbox=dict(
                                #         facecolor='white',
                                #         alpha=0.8,
                                #         boxstyle='square,pad=0.8'  # mismo boxstyle y padding que tenías
                                #     ),
                                #     verticalalignment='bottom',
                                #     horizontalalignment='left'
                                # )

                                # # ─── Ajuste de márgenes para maximizar espacio de ejes ─────────────────────────────
                                # plt.subplots_adjust(
                                #     left=0.03,    # margen izquierdo muy pequeño
                                #     right=0.97,   # margen derecho muy pequeño
                                #     top=0.97,     # margen superior pequeño
                                #     bottom=0.15,  # deja espacio para el texto abajo
                                #     wspace=0.35,  # reduce el espacio horizontal entre columnas
                                #     hspace=0.25   # reduce el espacio vertical entre filas
                                # )

                                # # Guardar y cerrar
                                # plot_combined_filename = os.path.join(
                                #     output_dir,
                                #     f"plot_combined_iter_{iteration_counter:04d}_z0_{z0:.2f}_delta_{delta:.2f}"
                                #     f"_ax_{angle_x:.2f}_ay_{angle_y:.2f}_az_{angle_z:.2f}.png"
                                # )
                                # plt.savefig(plot_combined_filename, dpi=300, bbox_inches='tight')
                                # plt.close(fig)




                                # OTHER CALCULATIONS: 
                                # Format parameters with 4 significant figures in scientific notation
                                A_By_rounded = f"{A_By:.4e}"
                                B_By_rounded = f"{B_By:.4e}"
                                C_Bphi_rounded = f"{C_Bphi:.4e}"

                                # 0) Define symbolic expressions with formatted parameters
                                r, alpha = sp.symbols('r alpha')
                                Br_expr = model_Br((r, alpha), float(a_Br))  # Será cero
                                By_expr = sp.sympify(A_By_rounded) + sp.sympify(B_By_rounded) * r**2
                                Bphi_expr = sp.sympify(C_Bphi_rounded) * r

                                params_Br_fit = a_Br
                                params_By_fit = A_By, B_By
                                params_Bphi_fit = C_Bphi

                                # SAVING VARIABLES: 
                                best_combination = (z0, angle_x, angle_y, angle_z, delta)
                                quality_factors = (R2_B, R2_Bx, R2_By, R2_Bz, R2_avg)

                                # Plot 1: Oriented Flux Rope and intersection with plane y = 0
                                plot1_vars = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs)

                                # Plot 2: 3D Representation
                                plot2_vars = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)
                            
                                # Plot 3: Cross section and trajectory inside
                                plot3_vars = (x_ellipse_local, y_ellipse_local, x_local, y_local, x_inter_local, y_inter_local, h_line_local, v_line_local, a_local, b_local, x_ellipse_rotated, y_ellipse_rotated, x_traj_rotated, y_traj_rotated, x_inter_rotated, y_inter_rotated, h_line_rotated, v_line_rotated, chord_angle_right)

                                # Plot 4: Radial and angular values of the trajectory parametrization
                                plot4_vars = (x_local, r_vals, phi_vals)

                                # Plot 5: In-situ data in Local Cartesian coordinates vs original GSE exp
                                plot5_vars = (x_traj_GSE, B_GSE_exp_tot, Bx_GSE_exp, By_GSE_exp, Bz_GSE_exp, x_traj, Bx_Local_exp, By_Local_exp, Bz_Local_exp, B_Local_total_exp) 

                                # Plot 6: In-situ Cylindrical Components and fitted model
                                plot6_vars = (x_traj, B_total_exp_cyl, Br_exp, By_exp_cyl, Bphi_exp, B_vector, Br_vector, By_vector, Bphi_vector) 

                                # Plot 7: Fitted Local and GSE components
                                plot7_vars = (x_traj, B_vector, Bx_traj, By_traj_cartesian, Bz_traj, x_traj_GSE, B_total_GSE, Bx_GSE, By_GSE, Bz_GSE)


                                viz_3d_vars_opt = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled,
                                                scale_factor, a, Z_max_scaled, z_cut_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 
                                                X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)
                                
                                a_section, b_section = a_local, b_local

                                # Plot 1 auxiliar variables:
                                plot1_extra_vars = x_at_Z_max, x_at_Z_min, x_limits, z_limits


    progress_bar.progress(100)
    progress_text.text("Processing complete! ✅")


    # POST PROCESSING
    ### (A). Data extraction
    if best_R2 is not None and best_combination is not None:
        st.success(f"Best R2 found: {best_R2:.4f}")

        R2_B, R2_Bx, R2_By, R2_Bz, R2_avg = quality_factors
        z0, angle_x, angle_y, angle_z, delta = best_combination

        # Plot 1: Oriented Flux Rope and intersection with plane y = 0
        X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs = plot1_vars
        x_at_Z_max, x_at_Z_min, x_limits, z_limits = plot1_extra_vars

        # Plot 2: 3D Representation
        X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm =  plot2_vars

        # Plt 2B: 3D Interactive Plot
        viz_3d_vars_opt = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled,
                                            scale_factor, a, Z_max_scaled, z_cut_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 
                                            X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)

        # Plot 3: Cross section and trajectory inside
        x_ellipse_local, y_ellipse_local, x_local, y_local, x_inter_local, y_inter_local, h_line_local, v_line_local, a_local, b_local, x_ellipse_rotated, y_ellipse_rotated, x_traj_rotated, y_traj_rotated, x_inter_rotated, y_inter_rotated, h_line_rotated, v_line_rotated, chord_angle_right = plot3_vars

        # Plot 4: Radial and angular values of the trajectory parametrization
        x_local, r_vals, phi_vals = plot4_vars

        # Plot 5: In-situ data in Local Cartesian coordinates vs original GSE exp
        x_traj_GSE, B_GSE_exp_tot, Bx_GSE_exp, By_GSE_exp, Bz_GSE_exp, x_traj, Bx_Local_exp, By_Local_exp, Bz_Local_exp, B_Local_total_exp = plot5_vars 

        # Plot 6: In-situ Cylindrical Components and fitted model
        x_traj, B_total_exp_cyl, Br_exp, By_exp_cyl, Bphi_exp, B_vector, Br_vector, By_vector, Bphi_vector = plot6_vars  

        # Plot 7: Fitted Local and GSE components
        x_traj, B_vector, Bx_traj, By_traj_cartesian, Bz_traj, x_traj_GSE, B_total_GSE, Bx_GSE, By_GSE, Bz_GSE = plot7_vars 

        a_Br = params_Br_fit 
        A_By, B_By = params_By_fit 
        C_Bphi = params_Bphi_fit 

    
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # ----------------------------------------------------------------------------------
        # PART A) FLUX ROPE FITTING
        # ----------------------------------------------------------------------------------
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        st.header("Flux Rope Fitting")

        # ----------------------------------------------------------------------------------
        # Plot 1) Oriented Flux Rope and intersection with plane y = 0
        # ----------------------------------------------------------------------------------
        st.subheader("1) Geometry of the fitted Flux Rope")


        # Calcular Z_max, Z_min y sus correspondientes x
        Z_max = np.max(Z_ellipse_scaled)
        Z_min = np.min(Z_ellipse_scaled)
        idx_max = np.argmax(Z_ellipse_scaled)
        idx_min = np.argmin(Z_ellipse_scaled)
        x_at_Z_max = X_ellipse_scaled[idx_max]
        x_at_Z_min = X_ellipse_scaled[idx_min]

        # Determinar la posición de la línea vertical según z_cut_scaled
        if z_cut_scaled > 0:
            x_vertical = x_at_Z_max
        else:
            x_vertical = x_at_Z_min

        # Límites de la línea vertical (usamos z_limits para cubrir toda la elipse)
        z_vertical_limits = [z_limits[0], z_limits[1]]


        # Create figure with two subplots: one 3D and one 2D
        fig = plt.figure(figsize=(10, 5))

        # 3D plot (left)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X_rot_scaled, Y_rot_scaled, Z_rot_scaled, color='lightblue', alpha=0.4)
        ax1.plot(X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled, 'r', label="Scaled ellipse")
        ax1.scatter(X_intersections_scaled, np.zeros_like(X_intersections_scaled), Z_intersections_scaled, color='red', s=50, label="Scaled intersection")
        ax1.plot(xs, ys, zs, 'g-', linewidth=2, label="Satellite trajectory")
        y_plane = np.zeros((10, 10))
        x_plane = np.linspace(x_limits[0], x_limits[1], 10)
        z_plane = np.linspace(z_limits[0], z_limits[1], 10)
        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        ax1.plot_surface(X_plane, y_plane, Z_plane, color='gray', alpha=0.2)
        ax1.plot(x_z_zero_limits, [0, 0], [0, 0], 'k--', linewidth=1.5, label="z = 0")
        ax1.plot([x_vertical, x_vertical], [0, 0], z_vertical_limits, 'b--', linewidth=1.5, label="Z max/min")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Rotated cylinder with satellite trajectory")
        ax1.legend()
        set_axes_equal(ax1)

        # 2D plot (right): Elliptical section in X vs Z
        ax2 = fig.add_subplot(122)
        ax2.plot(X_ellipse_scaled, Z_ellipse_scaled, 'r', label="Scaled ellipse")
        ax2.scatter(X_intersections_scaled, Z_intersections_scaled, color='red', s=50, label="Scaled intersection")
        ax2.plot(xs, zs, 'g-', linewidth=2, label="Satellite trajectory")
        ax2.plot(x_z_zero_limits, [0, 0], 'k--', linewidth=1.5, label="z = 0")
        ax2.plot([x_vertical, x_vertical], z_vertical_limits, 'b--', linewidth=1.5, label="Z max/min")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title("Elliptical section with horizontal cut")
        ax2.set_xlim(x_limits)
        ax2.set_ylim(z_limits)
        ax2.legend()
        ax2.set_aspect('equal')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        axis_cylinder_norm = axis_cylinder / np.linalg.norm(axis_cylinder)
        st.markdown(f"**Axis vector**: ({axis_cylinder_norm[0]:.3f}, {axis_cylinder_norm[1]:.3f}, {axis_cylinder_norm[2]:.3f})")

        # Compute the angle (radians) between the vector and the XY plane
        theta_xy = np.arctan2(axis_cylinder_norm[2],np.hypot(axis_cylinder_norm[0],  axis_cylinder_norm[1]))
        theta_xy_deg = np.degrees(theta_xy)

        phi_xz = np.arctan2(axis_cylinder_norm[1],np.hypot(axis_cylinder_norm[0],  axis_cylinder_norm[2]))
        phi_xz_deg = np.degrees(phi_xz)

        # using Unicode Greek letters θ (theta) and φ (phi):
        st.markdown(f"**θ (angle with the XY plane)**: {theta_xy_deg:.2f}°")
        st.markdown(f"**φ (angle with the XZ plane)**: {phi_xz_deg:.2f}°")


        # ----------------------------------------------------------------------------------
        # Plot 2) Plot 2: 3D Representation
        # ----------------------------------------------------------------------------------
        st.subheader("2) 3D Representation of the Flux Rope")

        # Create a Plotly figure
        fig = go.Figure()

        # 1. Surface of the cylinder
        fig.add_trace(go.Surface(
            x=X_rot_scaled,
            y=Y_rot_scaled,
            z=Z_rot_scaled,
            colorscale='Blues',
            opacity=0.4,
            showscale=False,
            name="Cylinder"
        ))

        # 2. Rescaled Ellipse
        fig.add_trace(go.Scatter3d(
            x=X_ellipse_scaled,
            y=np.zeros_like(X_ellipse_scaled),
            z=Z_ellipse_scaled,
            mode='lines',
            line=dict(color='red', width=2),
            name="Rescaled Ellipse"
        ))

        # 3. Rescaled Intersection points
        fig.add_trace(go.Scatter3d(
            x=X_intersections_scaled,
            y=np.zeros_like(X_intersections_scaled),
            z=Z_intersections_scaled,
            mode='markers',
            marker=dict(size=5, color='red'),
            name="Rescaled Intersection"
        ))

        # 4. Gray plane (X-Z plane at y=0)
        x_plane = np.linspace(- a_local * 1.5, a_local * 1.5, 10)
        z_plane = np.linspace(-a_local * 1.5, a_local * 1.5, 10)
        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        Y_plane = np.zeros((10, 10))
        fig.add_trace(go.Surface(
            x=X_plane,
            y=Y_plane,
            z=Z_plane,
            colorscale='Greys',
            opacity=0.2,
            showscale=False,
            name="Gray Plane"
        ))

        # 5. Intersection Line
        fig.add_trace(go.Scatter3d(
            x=[x1_scaled, x2_scaled],
            y=[0, 0],
            z=[z_cut_scaled, z_cut_scaled],
            mode='lines',
            line=dict(color='blue', width=4),
            name="Intersection Line"
        ))

        # 6. Extended Dashed Line
        x_extended = np.linspace(-a_local * 1.0, a_local * 1.0, 100)
        z_extended = np.full_like(x_extended, z_cut_scaled)
        fig.add_trace(go.Scatter3d(
            x=x_extended,
            y=np.zeros_like(x_extended),
            z=z_extended,
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name="Extended Dashed Line"
        ))

        # 7. Projected Ellipse
        fig.add_trace(go.Scatter3d(
            x=X_proj_ellipse,
            y=Y_proj_ellipse,
            z=Z_proj_ellipse,
            mode='lines',
            line=dict(color='green', width=2),
            name="Projected Ellipse"
        ))

        # 8. Projected Intersection points
        fig.add_trace(go.Scatter3d(
            x=X_proj_inter,
            y=Y_proj_inter,
            z=Z_proj_inter,
            mode='markers',
            marker=dict(size=5, color='green'),
            name="Projected Intersection"
        ))

        # 9. Projected Trajectory
        fig.add_trace(go.Scatter3d(
            x=X_proj_traj,
            y=Y_proj_traj,
            z=Z_proj_traj,
            mode='lines',
            line=dict(color='magenta', width=4),
            name="Projected Trajectory"
        ))

        # 10. Transverse Plane
        # Center of the square (point on the cylinder axis)
        center = np.array([0, 0, d / axis_cylinder_norm[2]])  # Adjust based on your axis and d

        # Find two orthogonal vectors in the plane perpendicular to axis_cylinder_norm
        norm = np.array(axis_cylinder_norm)
        # Choose an arbitrary vector not parallel to norm
        if abs(norm[0]) > abs(norm[2]):
            u = np.cross(norm, [0, 0, 1])  # Cross product with [0,0,1]
        else:
            u = np.cross(norm, [1, 0, 0])  # Cross product with [1,0,0]
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(norm, u)  # Second vector perpendicular to norm and u
        v = v / np.linalg.norm(v)  # Normalize

        # Define the square's corners in the plane
        s = 3 * a_local / 2  # Half the side length
        t = np.linspace(-s, s, 10)  # Grid points
        T1, T2 = np.meshgrid(t, t)
        # Parametric equation: center + t1*u + t2*v
        X_trans = center[0] + T1 * u[0] + T2 * v[0]
        Y_trans = center[1] + T1 * u[1] + T2 * v[1]
        Z_trans = center[2] + T1 * u[2] + T2 * v[2]

        fig.add_trace(go.Surface(
            x=X_trans,
            y=Y_trans,
            z=Z_trans,
            colorscale='YlOrBr',
            opacity=0.2,
            showscale=False,
            name="Transverse Plane"
        ))

        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data',  # Ensures equal aspect ratio
            ),
            title="Cylinder with Cut, Projection, and Trajectory",
            showlegend=True,
            height=800
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        # ----------------------------------------------------------------------------------
        # Plot 3) Cross section and trajectory inside
        # ----------------------------------------------------------------------------------

        st.subheader("3) Cross section and trajectory inside")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # LEFT SUBPLOT: original local coords
        # ------------------------------------------------------------------------------
        ax1.plot(x_ellipse_local, y_ellipse_local, 'r-', label="Elliptical Section (Local)")
        ax1.plot(x_local, y_local, 'b-', linewidth=2, label="Projected Trajectory")
        ax1.scatter(x_inter_local, y_inter_local, color='blue', s=50, label="Intersection Points")

        # Dashed chords
        ax1.plot(h_line_local[:, 0], h_line_local[:, 1], 'k--', linewidth=1)
        ax1.plot(v_line_local[:, 0], v_line_local[:, 1], 'k--', linewidth=1)

        # Capture print output for the left subplot angles
        left_angles = []
        for i, (xi, yi) in enumerate(zip(x_inter_local, y_inter_local)):
            # Grey line from center to intersection
            if i == 0:
                ax1.plot([0, xi], [0, yi], color='grey', linestyle='-', linewidth=1,
                        label="Center to Intersection")
            else:
                ax1.plot([0, xi], [0, yi], color='grey', linestyle='-', linewidth=1)
            
            # Angle from +X axis (local coords)
            angle_deg_left = np.degrees(np.arctan2(yi, xi)) % 360
            left_angles.append(angle_deg_left)
            
            # Arc from 0 deg up to angle_deg_left
            r_arc = 0.1 * np.sqrt(xi**2 + yi**2)  # radius ~10% of the line length
            arc_patch_left = Arc(
                (0, 0), 2*r_arc, 2*r_arc,
                angle=0,
                theta1=0,
                theta2=angle_deg_left,
                color='grey', linewidth=1
            )
            ax1.add_patch(arc_patch_left)
            
            # Label near the midpoint of the arc
            theta_mid = np.radians(0.5 * angle_deg_left)
            rx = 1.15 * r_arc * np.cos(theta_mid)
            ry = 1.15 * r_arc * np.sin(theta_mid)
            ax1.text(rx, ry, f"{angle_deg_left:.1f}°", color='black', fontsize=9)

        # --- Add labels for the ellipse's major and minor axes using scientific notation ---
        ax1.text(a_local, 0, f"  a = {a_local:.2e}", color='black', fontsize=10,
                horizontalalignment='left', verticalalignment='center')
        ax1.text(0, b_local, f"  b = {b_local:.2e}", color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='bottom')

        ax1.set_title("Elliptical Section in Local Coordinates")
        ax1.set_xlabel("Local X [km]")
        ax1.set_ylabel("Local Y [km]")
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend()

        # RIGHT SUBPLOT: rotated ellipse + horizontal trajectory
        # ------------------------------------------------------------------------------
        ax2.plot(x_ellipse_rotated, y_ellipse_rotated, 'r-', label="Rotated Ellipse")
        ax2.plot(x_traj_rotated, y_traj_rotated, 'b-', linewidth=2, label="Horizontal Trajectory")
        ax2.scatter(x_inter_rotated, y_inter_rotated, color='blue', s=50, label="Intersection Points")

        # Dashed chords
        ax2.plot(h_line_rotated[:, 0], h_line_rotated[:, 1], 'k--', linewidth=1)
        ax2.plot(v_line_rotated[:, 0], v_line_rotated[:, 1], 'k--', linewidth=1)

        for i, (xir, yir) in enumerate(zip(x_inter_rotated, y_inter_rotated)):
            # Grey line from center to intersection
            if i == 0:
                ax2.plot([0, xir], [0, yir], color='grey', linestyle='-', linewidth=1,
                        label="Center to Intersection")
            else:
                ax2.plot([0, xir], [0, yir], color='grey', linestyle='-', linewidth=1)
            
            # Angle in the rotated frame for the intersection
            angle_deg_right = np.degrees(np.arctan2(yir, xir)) % 360
            
            # We want an arc from the chord's angle to the intersection line's angle
            theta_start = chord_angle_right
            theta_end = angle_deg_right
            # Ensure the arc goes in the correct direction (counterclockwise)
            if theta_end < theta_start:
                theta_end += 360
                        
            # Arc radius ~10% of the intersection line length
            r_arc = 0.1 * np.sqrt(xir**2 + yir**2)
            arc_patch_right = Arc(
                (0, 0), 2*r_arc, 2*r_arc,
                angle=0,
                theta1=theta_start,
                theta2=theta_end,
                color='grey', linewidth=1
            )
            ax2.add_patch(arc_patch_right)
            
            # Place label near midpoint of that arc
            theta_mid = 0.5 * (theta_start + theta_end)
            theta_mid_rad = np.radians(theta_mid)
            rx = 1.15 * r_arc * np.cos(theta_mid_rad)
            ry = 1.15 * r_arc * np.sin(theta_mid_rad)
            
            # Label with the difference in angles (theta_end - theta_start)
            arc_span = theta_end - theta_start
            ax2.text(rx, ry, f"{arc_span:.1f}°", color='black', fontsize=9)

        ax2.set_title("Rotated Ellipse with Horizontal Cut")
        ax2.set_xlabel("Rotated Local X [km]")
        ax2.set_ylabel("Rotated Local Y [km]")
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()

        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Close the figure to free memory
        plt.close(fig)

        # ----------------------------------------------------------------------------------
        # Plot 4) Radial and angular values of the trajectory parametrization
        # ----------------------------------------------------------------------------------

        st.subheader("4) Radial and angular values of the trajectory parametrization")

        # cálculo original
        r_vals, phi_vals = elliptical_coords(x_local, y_local, xc, zc, a_local, b_local, theta)

        # unwrap para eliminar saltos de 2π
        phi_vals = np.unwrap(phi_vals)

        # a continuación lo conviertes a grados
        phi_vals_deg = np.degrees(phi_vals)


        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left Subplot: r_vals and phi_vals vs. x_local ---
        darker_orange = '#e69500'  # Hex code for a darker orange
        ax1.plot(x_local, r_vals, color=darker_orange, label='r_vals', linewidth=2)
        ax1.set_xlabel('x_local')
        ax1.set_ylabel('r_vals', color=darker_orange)
        ax1.tick_params(axis='y', labelcolor=darker_orange)
        ax1.grid(True)

        # Create a second y-axis for phi_vals on the left subplot (still purple)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_local, phi_vals_deg, color='purple', label='phi_vals (degrees)', linewidth=2)
        ax1_twin.set_ylabel('phi_vals (degrees)', color='purple')
        ax1_twin.tick_params(axis='y', labelcolor='purple')

        # Add title and legends for the left subplot
        ax1.set_title('r_vals and phi_vals vs. x_local')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # --- Right Subplot: Elliptical Section in Local Coordinates ---
        ax2.plot(x_ellipse_local, y_ellipse_local, 'r-', label="Elliptical Section (Local)")
        ax2.plot(x_local, y_local, 'b-', linewidth=2, label="Projected Trajectory")
        ax2.scatter(x_inter_local, y_inter_local, color='blue', s=50, label="Intersection Points")

        # Dashed chords
        ax2.plot(h_line_local[:, 0], h_line_local[:, 1], 'k--', linewidth=1)
        ax2.plot(v_line_local[:, 0], v_line_local[:, 1], 'k--', linewidth=1)

        # Add labels for the ellipse's major and minor axes using scientific notation
        ax2.text(a_local, 0, f"  a = {a_local:.2e}", color='black', fontsize=10,
                horizontalalignment='left', verticalalignment='center')
        ax2.text(0, b_local, f"  b = {b_local:.2e}", color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='bottom')

        # Set title, labels, and other properties for the right subplot
        ax2.set_title("Elliptical Section in Local Coordinates")
        ax2.set_xlabel("Local X [km]")
        ax2.set_ylabel("Local Y [km]")
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Close the figure to free memory
        plt.close(fig)


        # ----------------------------------------------------------------------------------
        # Plot 5) In-situ data in Local Cartesian coordinates vs original GSE exp
        # ----------------------------------------------------------------------------------

        st.subheader("5) In-Situ Local and GSE Data")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        x_traj_GSE = x_traj_GSE[::-1]  # Reverse the order of x_traj_GSE for plotting
        # --- Plot GSE magnetic field components in local Cartesian coordinates ---
        ax1.plot(x_traj_GSE, B_GSE_exp_tot, 'k-', label=r'$|\mathbf{B}|$')
        ax1.plot(x_traj_GSE, Bx_GSE_exp, 'b-', label=r'$B_x$')
        ax1.plot(x_traj_GSE, By_GSE_exp, 'r-', label=r'$B_y$')
        ax1.plot(x_traj_GSE, Bz_GSE_exp, 'g-', label=r'$B_z$')
        ax1.set_xlabel("X GSE rotated")
        ax1.set_ylabel("B components in GSE (nT)")
        ax1.set_title("B components in GSE Reference System")
        ax1.legend()
        ax1.grid(True)
        x_traj_GSE = x_traj_GSE[::-1]  # Reverse the order of x_traj_GSE for plotting

        # --- Plot LOCAL magnetic field components in GSE coordinates ---
        ax2.plot(x_traj, Bx_Local_exp, 'b-', label=r"$B_x^L$")
        ax2.plot(x_traj, By_Local_exp, 'r-', label=r"$B_y^L$")
        ax2.plot(x_traj, Bz_Local_exp, 'g-', label=r"$B_z^L$")
        ax2.plot(x_traj, B_Local_total_exp, 'k--', label=r"$|\mathbf{B}|$")
        ax2.set_xlabel("X Local axis")
        ax2.set_ylabel("Magnetic Field Value (nT)")
        ax2.set_title("B components in Local Cartesian Reference System")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)
        
        # ----------------------------------------------------------------------------------
        # Plot 6) In-situ Cylindrical Components and fitted model
        # ----------------------------------------------------------------------------------

        st.subheader("6) In-Situ and Fitted Cylindrical Components")

        # Crear un único plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- Datos exportados (líneas continuas) ---
        ax.plot(x_traj, B_total_exp_cyl,   'k-', label=r'$\mathrm{exp}:|\mathbf{B}|$')
        ax.plot(x_traj, Br_exp,            'b-', label=r'$\mathrm{exp}:B^r$')
        ax.plot(x_traj, By_exp_cyl,        'r-', label=r'$\mathrm{exp}:B^y$')
        ax.plot(x_traj, Bphi_exp,          'g-', label=r'$\mathrm{exp}:B^\varphi$')

        # --- Datos ajustados (líneas discontinuas más gruesas) ---
        ax.plot(x_traj, B_vector,          'k--', linewidth=2.0, label=r'$\mathrm{fit}:|\mathbf{B}|$')
        ax.plot(x_traj, Br_vector,         'b--', linewidth=2.0, label=r'$\mathrm{fit}:B^r$')
        ax.plot(x_traj, By_vector,         'r--', linewidth=2.0, label=r'$\mathrm{fit}:B^y$')
        ax.plot(x_traj, Bphi_vector,       'g--', linewidth=2.0, label=r'$\mathrm{fit}:B^\varphi$')

        # Etiquetas y ajustes
        ax.set_xlabel("X local rotated")
        ax.set_ylabel("Magnetic Field (Cylindrical Components)")
        ax.set_title("Experimental vs Fitted in Cylindrical Coordinates")
        ax.grid(True)
        ax.legend(loc='upper right', ncol=2, fontsize='small')

        plt.tight_layout()
        st.pyplot(fig)


        # ----------------------------------------------------------------------------------
        # Plot 7) Magnetic field representation in the cross section
        # ----------------------------------------------------------------------------------

        # --- Fitted magnetic field models ---
        def Br_model_fitted(r, alpha):
            return np.zeros_like(r)  # Radial component (always zero)

        def By_model_fitted(r, alpha):
            r = r * a_local * 10**3
            return A_By + B_By * r**2  # Y component with fitted parameters

        def Bphi_model_fitted(r, alpha):
            r = r * a_local * 10**3
            return C_Bphi * r  # Azimuthal component with fitted parameter

        # --- Create the mesh for the cross section ---
        Npt = 300
        x_ellipse_rotated = x_ellipse_rotated * 10**3
        y_ellipse_rotated = y_ellipse_rotated * 10**3
        x_min = min(x_ellipse_rotated) - 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
        x_max = max(x_ellipse_rotated) + 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
        y_min = min(y_ellipse_rotated) - 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
        y_max = max(y_ellipse_rotated) + 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
        X_grid, Y_grid = np.meshgrid(
            np.linspace(x_min, x_max, Npt),
            np.linspace(y_min, y_max, Npt)
        )

        # --- Parameters of the rotated ellipse ---
        xc = np.mean(x_ellipse_rotated)
        yc = np.mean(y_ellipse_rotated)

        # Estimate semi-axes and rotation angle
        ellipse_local_2d_rotated = np.column_stack((x_ellipse_rotated - xc, y_ellipse_rotated - yc))
        ellipse_model = EllipseModel()
        ellipse_model.estimate(ellipse_local_2d_rotated)
        _, _, a_ellipse, b_ellipse, theta_ellipse = ellipse_model.params
        a_ell = 1  # Normalized scale

        # --- Calculate elliptical coordinates ---
        r_grid, alpha_grid = elliptical_coords(
            X_grid, Y_grid, xc, yc,
            a_ellipse, b_ellipse, theta_ellipse
        )

        # --- Evaluate fitted magnetic field components on the mesh ---
        Br_vals_fitted   = Br_model_fitted(r_grid, alpha_grid)
        By_vals_fitted   = By_model_fitted(r_grid, alpha_grid)
        Bphi_vals_fitted = Bphi_model_fitted(r_grid, alpha_grid)

        # --- Metric components on the 2D mesh ---
        grr_corrected = a_ell**2 * (np.cos(alpha_grid)**2 + delta**2 * np.sin(alpha_grid)**2)
        gyy_corrected = np.ones_like(r_grid)
        gphiphi_corrected = a_ell**2 * r_grid**2 * (np.sin(alpha_grid)**2 + delta**2 * np.cos(alpha_grid)**2)
        grphi_corrected = a_ell**2 * r_grid * np.sin(alpha_grid) * np.cos(alpha_grid) * (delta**2 - 1)


        # --- Total magnitude of the fitted magnetic field on the mesh ---
        B_total_fitted = np.sqrt(
            grr_corrected * Br_vals_fitted**2 +
            gyy_corrected * By_vals_fitted**2 +
            gphiphi_corrected * Bphi_vals_fitted**2 +
            2 * grphi_corrected * Br_vals_fitted * Bphi_vals_fitted
        )

        # --- Mask points outside the ellipse (r > 1) ---
        mask = (r_grid > 1.0)
        Br_masked_fitted   = np.ma.array(Br_vals_fitted,   mask=mask)
        By_masked_fitted   = np.ma.array(By_vals_fitted,   mask=mask)
        Bphi_masked_fitted = np.ma.array(Bphi_vals_fitted, mask=mask)
        Btotal_masked_fitted = np.ma.array(B_total_fitted, mask=mask)

        # --- 2D contour plots in Streamlit (2x2) ---
        st.subheader("7) Magnetic Field Cross Section (Fitted)")

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2 filas, 2 columnas
        fields_fitted = [Br_masked_fitted, By_masked_fitted, Bphi_masked_fitted, Btotal_masked_fitted]
        titles = [r"$B_r$", r"$B_y$", r"$B_{\phi}$", r"$|\mathbf{B}|$"]

        x_traj_rotated, y_traj_rotated = x_traj_rotated * 10**3, y_traj_rotated * 10**3
        
        # Flatten axs for easy iteration
        axs_flat = axs.flatten()

        # Define a consistent colormap
        cmap = plt.get_cmap('viridis')  # Low values purple, high values yellow

        for ax, field, title in zip(axs_flat, fields_fitted, titles):
            # Plot the ellipse boundary and trajectory first
            ax.plot(x_ellipse_rotated, y_ellipse_rotated, 'k-', lw=2, label="Ellipse Boundary")
            ax.plot(x_traj_rotated, y_traj_rotated, 'b--', lw=2, label="Trajectory")
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("X local rotated")
            ax.set_ylabel("Y local rotated")
            ax.set_title(f"Fitted: {title}", fontsize=15)

            # Check if the field is fully masked or constant zero (e.g., Br)
            if np.all(field.mask) or np.all(field[~field.mask] == 0):
                # For zero fields like Br, display text and reserve colorbar space
                ax.text(0.5, 0.5, f"{title} = 0", transform=ax.transAxes, ha='center', va='center', fontsize=12)
                # Create a dummy contour to reserve colorbar space
                dummy_data = np.zeros_like(X_grid)
                ctf = ax.contourf(X_grid, Y_grid, dummy_data, levels=[-1e-30, 1e-30], cmap=cmap, alpha=0)
                # Add an invisible colorbar to maintain layout
                cbar = fig.colorbar(ctf, ax=ax, shrink=0.9)
                cbar.ax.set_visible(False)
                ax.legend()
                continue

            # Compute contour plot for non-zero fields (By, Bphi, |B|)
            ctf = ax.contourf(X_grid, Y_grid, field, 50, cmap=cmap)
            ax.contour(X_grid, Y_grid, field, 10, colors='k', alpha=0.4)
            # Add colorbar with default configuration
            cbar = fig.colorbar(ctf, ax=ax, shrink=0.9)
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Fitted formulas of the magnetic field components
        # ----------------------------------------------------------------------------------
        st.markdown("<h2 style='font-size:20px;'>Model used for each component of the magnetic field</h2>", unsafe_allow_html=True)
        st.latex(r"B_r = 0")
        st.latex(r"B_y(r) = A + B r^2")
        st.latex(r"B_\phi(r) = C r")

        st.markdown("<h2 style='font-size:20px;'>Resulting Formulas</h2>", unsafe_allow_html=True)
        st.latex(f"B_r = {sp.latex(Br_expr)}")
        st.latex(f"B_y(r) = {sp.latex(By_expr)}")
        st.latex(f"B_\\phi(r) = {sp.latex(Bphi_expr)}")
        st.latex(r"\nabla \cdot \mathbf{B} = 0")


        # ----------------------------------------------------------------------------------
        # Plot 8) Fitted Local and GSE components
        # ----------------------------------------------------------------------------------

        st.subheader("8) Fitted Local and GSE Components")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # --- Plot magnetic field components in local Cartesian coordinates ---
        ax1.plot(x_traj, B_vector, 'k-', label=r'$|\mathbf{B}|$')
        ax1.plot(x_traj, Bx_traj, 'b-', label=r'$B_x$')
        ax1.plot(x_traj, By_traj_cartesian, 'r-', label=r'$B_y$')
        ax1.plot(x_traj, Bz_traj, 'g-', label=r'$B_z$')
        ax1.set_xlabel("X local rotated")
        ax1.set_ylabel("Magnetic Field (Local Cartesian Components)")
        ax1.set_title("Local Coordinates")
        ax1.legend()
        ax1.grid(True) 

        # --- Plot magnetic field components in GSE coordinates ---
        ax2.plot(x_traj_GSE, B_total_GSE, 'k-', label=r'$|\mathbf{B}|$')
        ax2.plot(x_traj_GSE, Bx_GSE, 'b-', label=r'$B_{x,\mathrm{GSE}}$')
        ax2.plot(x_traj_GSE, By_GSE, 'r-', label=r'$B_{y,\mathrm{GSE}}$')
        ax2.plot(x_traj_GSE, Bz_GSE, 'g-', label=r'$B_{z,\mathrm{GSE}}$')
        ax2.plot(x_traj_GSE, B_total_GSE, 'k-', label=r'$|\mathbf{B}|$')
        ax2.plot(x_traj_GSE, Bx_GSE, 'b-', label=r'$B_{x,\mathrm{GSE}}$')
        ax2.plot(x_traj_GSE, By_GSE, 'r-', label=r'$B_{y,\mathrm{GSE}}$')
        ax2.plot(x_traj_GSE, Bz_GSE, 'g-', label=r'$B_{z,\mathrm{GSE}}$')
        ax2.set_xlabel("X GSE axis")
        ax2.set_ylabel("Magnetic Field (GSE Components)")
        ax2.set_title("GSE Coordinates")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)


        # ----------------------------------------------------------------------------------
        # Plot 9) Fitting to the original Data
        # ----------------------------------------------------------------------------------

        # Reverse the data arrays for comparison
        # B_vector = B_vector[::-1]
        # Bx_GSE   = Bx_GSE[::-1]
        # By_GSE   = By_GSE[::-1]
        # Bz_GSE   = Bz_GSE[::-1]
        adjusted_data = [B_vector, Bx_GSE, By_GSE, Bz_GSE]

        # --- Comparative plot in Streamlit ---
        st.subheader("9) Fitting to the original Data")
        fig_compare, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        components      = ['B', 'Bx', 'By', 'Bz']
        data_compare    = [B_data, Bx_data, By_data, Bz_data]  # Original data in GSE
        titles_compare  = [
            "Magnetic Field Intensity (B)",
            "Magnetic Field Component Bx",
            "Magnetic Field Component By",
            "Magnetic Field Component Bz",
        ]

        # Define start and end points of the segment
        start_segment = ddoy_data[initial_point - 1]
        end_segment   = ddoy_data[final_point - 2]

        for i, (component, orig_data, adj_data, title) in enumerate(
            zip(components, data_compare, adjusted_data, titles_compare)
        ):
            # Plot experimental data as black dots
            ax[i].scatter(ddoy_data, orig_data, color='black', s=10, label=f'{component} Original')
            
            # Plot fitted data as a red dashed line (if available)
            if adj_data is not None:
                ax[i].plot(
                    ddoy_data[initial_point - 1:final_point - 1],
                    adj_data,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'{component} Fitted'
                )
            
            # Vertical lines for the start and end of the segment
            ax[i].axvline(x=start_segment, color='gray', linestyle='--', label='Start of Segment')
            ax[i].axvline(x=end_segment,   color='gray', linestyle='--', label='End of Segment')
            
            # Subplot configuration
            ax[i].set_title(title, fontsize=18, fontweight='bold')
            ax[i].set_ylabel(f"{component} (nT)", fontsize=14)
            ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            ax[i].minorticks_on()
            ax[i].legend(fontsize=12)

        ax[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_compare)
        plt.close(fig_compare)

        # --- Fitting parameters results ---
        st.markdown("<h2 style='font-size:20px;'>Fitting Parameters</h2>", unsafe_allow_html=True)

        # Assuming z0, angle_x, angle_y, angle_z, and delta are defined
        # If not, adjust these values to your context
        st.latex(f"z_0 = {z0:.2f}")
        st.latex(f"\\theta_x = {np.rad2deg(angle_x):.2f}^\\circ")
        st.latex(f"\\theta_y = {np.rad2deg(angle_y):.2f}^\\circ")
        st.latex(f"\\theta_z = {np.rad2deg(angle_z):.2f}^\\circ")
        st.latex(f"\\delta = {delta:.2f}")

        # Display Goodness-of-Fit (R²) results in Streamlit
        st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results</h2>", unsafe_allow_html=True)
        st.latex(f"R^2_B = {R2_B:.4f}")
        st.latex(f"R^2_{{Bx}} = {R2_Bx:.4f}")
        st.latex(f"R^2_{{By}} = {R2_By:.4f}")
        st.latex(f"R^2_{{Bz}} = {R2_Bz:.4f}")
        st.latex(f"R^2_{{avg}} = {R2_avg:.4f}")

        # ----------------------------------------------------------------------------------
        # Plot 10) Current Density formulas and Plot in the Cross section
        # ----------------------------------------------------------------------------------

        # Custom LaTeX printer (unchanged)
        class CustomLatexPrinter(LatexPrinter):
            def _print_Float(self, expr):
                abs_expr = abs(float(expr))
                if abs_expr == 0:
                    return "0"
                
                exponent = int(sp.log(abs_expr, 10))
                mantissa = expr / (10 ** exponent)
                
                while abs(mantissa) >= 10:
                    mantissa /= 10
                    exponent += 1
                while abs(mantissa) < 1:
                    mantissa *= 10
                    exponent -= 1
                
                mantissa_str = "{:.4f}".format(mantissa).rstrip('0').rstrip('.')
                
                if -2 <= exponent <= 2:
                    original_number = expr
                    return "{:.4f}".format(original_number).rstrip('0').rstrip('.')
                
                return f"{mantissa_str} \\cdot 10^{{{exponent}}}"

        custom_latex = CustomLatexPrinter()

        # Current density calculations
        # Define symbolic variables (for symbolic expressions only)
        r, phi = sp.symbols('r phi')  # Use phi to match the provided notation
        mu_0_sym = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

        # Metric parameters (ensure these are numerical)
        # a_ell = a_local * 10**3 # Semi-major axis of the ellipse (in m)
        a = float(a_ell)  # Typically 1 in your code
        delta = float(delta)  # From your main loop

        # Map the fitted parameters to the new model (convert to numerical values)
        A = float(A_By) * 10**(-9)  # [T]
        B = float(B_By)  * 10**(-9)# [T/m^2]
        C = float(C_Bphi)/a_ell  * 10**(-9) # [T/m]
        mu_0 = float(mu_0_sym)

        # Define the current density expressions (symbolic, for display)
        mu_0_sym = sp.Float(mu_0_sym)  # Permeability of free space (for symbolic use)
        jr = 0  # j_r is zero

        # j_phi
        jphi = (2 * B) / (delta * mu_0_sym)

        # j_y, with the condition on delta
        if 0.999 <= float(delta) <= 1.001:
            jy = (-3 * C * r) / (delta * mu_0)
        else:
            jy = (-3 * C * r) / (delta * mu_0) * (sp.sin(phi)**2 + delta**2 * sp.cos(phi)**2)
            

        # Display the current density formulas with the specified dependencies
        st.write("## Current Density Formulas")

        # jr(r, phi)
        st.write("**Radial Current Density:**")
        st.latex(f"j_r(r, \\phi) = {custom_latex.doprint(jr)}")

        # jphi(r, phi)
        st.write("**Azimuthal Current Density:**")
        st.latex(f"j_\\phi(r, \\phi) = {custom_latex.doprint(jphi)}")

        # jy(r, phi)
        st.write("**Current Density in y:**")
        st.latex(f"j_y(r, \\phi) = {custom_latex.doprint(jy)}")

        # ----------------------------------------------------------------------------------
        # Plot: Current Density Representation in the Cross Section
        # ----------------------------------------------------------------------------------
        def jr_model_fitted(r, alpha):
            """
            Radial current density component (numerical evaluation)
            """
            return np.zeros_like(r)

        def jphi_model_fitted(r, alpha):
            """
            Azimuthal current density component (numerical evaluation)
            """
            return np.full_like(r, - (2 * B) / (delta * mu_0))

        def jy_model_fitted(r, alpha):
            """
            Y current density component (numerical evaluation)
            """
            if 0.999 <= float(delta) <= 1.001:
                return (-3 * C * r) / (delta * mu_0)
            else:
                return (-3 * C * r) / (delta * mu_0) * (np.sin(alpha)**2 + delta**2 * np.cos(alpha)**2)
                

        # Evaluate current density components on the grid
        # Convert coordinates from km to m (assuming x_ellipse_rotated is in km)
        r_grid_m = r_grid * 1e3  # Convert km to m

        jr_vals_fitted = jr_model_fitted(r_grid_m, alpha_grid)
        jphi_vals_fitted = jphi_model_fitted(r_grid_m, alpha_grid)
        jy_vals_fitted = jy_model_fitted(r_grid_m, alpha_grid)

        # Compute the total current density magnitude using metric coefficients
        j_total_fitted = np.sqrt(
            grr_corrected * jr_vals_fitted**2 +
            gyy_corrected * jy_vals_fitted**2 +
            gphiphi_corrected * jphi_vals_fitted**2 +
            2 * grphi_corrected * jr_vals_fitted * jphi_vals_fitted
        )
        

        # Mask points outside the ellipse (r > 1)
        mask = (r_grid > 1.0)
        jr_masked_fitted = np.ma.array(jr_vals_fitted, mask=mask)
        jy_masked_fitted = np.ma.array(jy_vals_fitted, mask=mask)
        jphi_masked_fitted = np.ma.array(jphi_vals_fitted, mask=mask)
        jtotal_masked_fitted = np.ma.array(j_total_fitted, mask=mask)

        # Create 2D contour plots in Streamlit (2x2)
        st.subheader("Current Density Cross Section (Fitted)")

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2 rows, 2 columns
        fields_fitted = [jr_masked_fitted, jy_masked_fitted, jphi_masked_fitted, jtotal_masked_fitted]
        titles = [r"$j_r$", r"$j_y$", r"$j_{\phi}$", r"$|\mathbf{j}|$"]

        # Flatten axs for easy iteration
        axs_flat = axs.flatten()

        # Print data ranges for debugging
        for i, (field, title) in enumerate(zip(fields_fitted, titles)):
            if np.all(field.mask):
                print(f"{title} is fully masked")
            else:
                valid_data = field[~field.mask]
                if valid_data.size > 0:
                    print(f"{title} range: min={np.min(valid_data):.2e}, max={np.max(valid_data):.2e}")
                else:
                    print(f"{title} has no valid data after masking")

        # Define a consistent colormap
        cmap = plt.get_cmap('viridis')  # Low values purple, high values yellow

        # Find the maximum absolute value across j_y and |j| for symmetric normalization
        max_abs_value = max(
            np.max(np.abs(jy_masked_fitted[~jy_masked_fitted.mask])),
            np.max(np.abs(jtotal_masked_fitted[~jtotal_masked_fitted.mask]))
        )

        for i, (ax, field, title) in enumerate(zip(axs_flat, fields_fitted, titles)):
            # Plot the ellipse boundary and trajectory first
            ax.plot(x_ellipse_rotated, y_ellipse_rotated, 'k-', lw=2, label="Ellipse Boundary")
            ax.plot(x_traj_rotated, y_traj_rotated, 'b--', lw=2, label="Trajectory")
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("X local rotated")
            ax.set_ylabel("Y local rotated")
            ax.set_title(f"Fitted: {title}", fontsize=15)

            # Check if the field is fully masked or constant zero
            if np.all(field.mask) or np.all(field[~field.mask] == 0):
                # For zero fields like j_r, display text and reserve colorbar space
                ax.text(0.5, 0.5, f"{title} = 0", transform=ax.transAxes, ha='center', va='center', fontsize=12)
                # Create a dummy contour to reserve colorbar space
                dummy_data = np.zeros_like(X_grid)
                ctf = ax.contourf(X_grid, Y_grid, dummy_data, levels=[-1e-30, 1e-30], cmap=cmap, alpha=0)
                # Add an invisible colorbar to maintain layout
                cbar = fig.colorbar(ctf, ax=ax, shrink=0.9, label='Current Density (A/m²)', format='%.2e')
                cbar.ax.set_visible(False)
                ax.legend()
                continue

            # Compute min and max for valid (unmasked) data
            valid_data = field[~field.mask]
            if valid_data.size == 0:
                # No valid data after masking
                ax.text(0.5, 0.5, f"{title} = No Data", transform=ax.transAxes, ha='center', va='center', fontsize=12)
                dummy_data = np.zeros_like(X_grid)
                ctf = ax.contourf(X_grid, Y_grid, dummy_data, levels=[-1e-30, 1e-30], cmap=cmap, alpha=0)
                cbar = fig.colorbar(ctf, ax=ax, shrink=0.9, label='Current Density (A/m²)', format='%.2e')
                cbar.ax.set_visible(False)
                ax.legend()
                continue

            # Use symmetric normalization centered at zero for j_y and |j|
            if title in [r"$j_y$", r"$|\mathbf{j}|$"]:
                vmin = -max_abs_value
                vmax = max_abs_value
            else:
                # For j_phi or other fields, use their actual range
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
                # Handle constant fields
                if vmin == vmax:
                    if abs(vmin) < 1e-30:  # Effectively zero
                        vmin, vmax = -1e-30, 1e-30
                    else:
                        # Expand range slightly for non-zero constant fields (e.g., j_phi)
                        scale = abs(vmin) * 0.001 if abs(vmin) > 1e-10 else 1e-30
                        vmin = vmin - scale
                        vmax = vmax + scale

            # Create normalization
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            # Create contour plot with consistent colormap and normalization
            ctf = ax.contourf(X_grid, Y_grid, field, 50, cmap=cmap, norm=norm)
            ax.contour(X_grid, Y_grid, field, 10, colors='k', alpha=0.4)
            # Add colorbar with scientific notation formatting
            cbar = fig.colorbar(ctf, ax=ax, shrink=0.9, label='Current Density (A/m²)', format='%.2e')
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)





        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # ----------------------------------------------------------------------------------
        # PART B) CME Propagation
        # ----------------------------------------------------------------------------------
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        st.header("CME Propagation")

        # ----------------------------------------------------------------------------------
        # Plot 11) Propagation along the Interplanetary Medium
        # ----------------------------------------------------------------------------------
        st.subheader("11) CME Propagation Video")

        # 11.A) Physical Parameters from the Fitting
        # ----------------------------------------------------------------------------------
        # Astronomical constants
        AU = 149597870.7 * 1000  # [m] 1 AU in meters
        AU_km = 149597870.7      # [km] 1 AU in kilometers
        R_s = 6.96e8             # [m] Solar radius
        R_s_km = R_s / 1000      # [km] Solar radius

        # CME dimensions at the source (Sun)
        b_max_0 = 0.1 * R_s_km   # [km] Minor radius at the Sun
        a_max_0 = b_max_0 / delta  # [km] Major radius at the Sun
        R_0 = 4 * a_max_0        # [km] Major radius of the torus
        r_start = 1.8 * R_s      # [m] Starting distance for integration (1.8 R_s)
        r0 = 2.5 * R_s           # [m] Reference point r0 (2.5 R_s)
        r0_km = r0 / 1000        # [km]

        # CME dimensions at detection point
        r1 = distance * AU       # [m] Distance to detection point
        b_1 = b_section          # [km] Minor radius at detection
        a_1 = b_1 / delta        # [km] Major radius at detection
        b_1_m = b_1 * 1000       # [m]
        a_1_m = a_1 * 1000       # [m]

        # CME volume and mass
        Vol1 = (np.pi**2 * r1 * a_1_m * b_1_m) / 2  # [m^3]
        m_cme = density_avg * Vol1                  # [kg] Using provided density_avg
        Vol1_AU = Vol1 / (AU**3)                    # [AU^3]

        # Other distances
        r2 = 1.0 * AU            # [m] Forward integration point (1 AU)
        r3 = 1.524 * AU          # [m] Average orbital distance of Mars (1.524 AU)
        r_max = 4.0 * AU         # [m] Maximum distance for forward integration
        x_end = 1.0 * AU_km      # [km] End distance for animation set to 1 AU

        # CME properties
        v1 = Vsw * 10**3         # [m/s] CME velocity at detection
        vector_director = axis_cylinder_norm  # CME axis direction

        # Orbital parameters of planets
        a_earth = AU_km
        e_earth = 0.0167
        b_earth = a_earth * np.sqrt(1 - e_earth**2)
        a_mercury = 0.387 * AU_km
        e_mercury = 0.2056
        b_mercury = a_mercury * np.sqrt(1 - e_mercury**2)
        a_venus = 0.723 * AU_km
        e_venus = 0.0068
        b_venus = a_venus * np.sqrt(1 - e_venus**2)
        a_mars = 1.524 * AU_km  # Average orbit of Mars in km
        e_mars = 0.0934         # Eccentricity of Mars' orbit
        b_mars = a_mars * np.sqrt(1 - e_mars**2)
        satellite_position = np.array([a_earth, 0, 0])  # [km]

        # Dynamics of the CME equation parameters
        GM = 1.32712440018e20    # [m^3/s^2] Gravitational constant * solar mass
        C_d = 1.0                # Drag coefficient
        dotM = 1e9               # [kg/s] Solar wind mass loss rate
        c_s = 120e3              # [m/s] Sound speed
        r_c = 5 * R_s            # [m] Critical radius for solar wind

        # Simulation configuration
        num_frames = n_frames          # Number of animation frames

        fps = 5 + (60 - 5) * (num_frames - 5) / (1000 - 5)
        # nos aseguramos de que quede dentro de [min_fps, max_fps] y entero
        fps = int(max(5, min(fps, 60)))

        # 11.B) Find the cross section that aligns with the CME axis
        # ----------------------------------------------------------------------------------
        # CME Rotation Parameters
        if vector_director[0] < 0:
            vector_director = -vector_director
        vector_director = vector_director / np.linalg.norm(vector_director)
        if vector_director[1] != 0:
            theta_x = np.arctan2(vector_director[2], vector_director[1])
        else:
            theta_x = np.pi/2 if vector_director[2] > 0 else -np.pi/2
        theta_x_deg = np.degrees(theta_x)

        if vector_director[1] != 0:
            tan_phi0 = -vector_director[0] * np.cos(theta_x) / vector_director[1]
            phi0 = np.arctan(tan_phi0)
        else:
            phi0 = -np.pi/2 if vector_director[0] > 0 else np.pi/2
        if phi0 > np.pi/2:
            phi0 -= np.pi
        elif phi0 < -np.pi/2:
            phi0 += np.pi
        phi0_deg = np.degrees(phi0)

        b0_m = b_max_0 * 1000  # [m]
        expansion_ratio = np.log(b_1 / b_max_0) / np.log(r1 / r_start)

        # 11.C) Define Functions to solve the CME dynamics
        # ----------------------------------------------------------------------------------
        def v_sw_near(r):
            return c_s * np.exp(2.0 * np.log(r / r_c) + 2.0 * (r_c / r) - 1.5)

        def v_sw_far(r):
            inside = 4.0 * np.log(r / r_c) + 4.0 * (r_c / r) - 3.0
            return c_s * np.sqrt(np.maximum(inside, 0.0))

        def v_sw(r):
            transition = 1.0 / (1.0 + np.exp(-10.0 * (r - r_c) / r_c))
            return (1 - transition) * v_sw_near(r) + transition * v_sw_far(r)

        def density(r):
            return dotM / (4.0 * np.pi * r**2 * v_sw(r))

        def A(r):
            return np.pi * delta * b0_m**2 * (r / r_start)**(2 * expansion_ratio)

        def dvdr(r, v):
            if v <= 0:
                return 0.0
            drag = (C_d * density(r) * A(r) / m_cme) * (v - v_sw(r)) * np.abs(v - v_sw(r))
            return (-GM / r**2 - drag) / v

        # 11.D) Integrate Velocity
        # ----------------------------------------------------------------------------------
        # Backward integration from r1 to r_start (1.8 R_s)
        sol_backward = solve_ivp(dvdr, (r1, r_start), [v1], dense_output=False, max_step=1e9)
        r_backward = sol_backward.t[::-1]
        v_backward = sol_backward.y[0][::-1]

        # Forward integration from r1 to r_max
        sol_forward = solve_ivp(dvdr, (r1, r_max), [v1], dense_output=False, max_step=1e9)
        r_forward = sol_forward.t
        v_forward = sol_forward.y[0]

        # Merge trajectories
        r_full = np.concatenate((r_backward, r_forward[1:]))
        v_full = np.concatenate((v_backward, v_forward[1:]))

        # Ensure sorted order
        idx_sort = np.argsort(r_full)
        r_full = r_full[idx_sort]
        v_full = v_full[idx_sort]

        r_full_km = r_full / 1000
        v_full_kms = v_full / 1000

        # Compute travel times
        # Backward: r_start to r0 (2.5 R_s)
        mask_start_r0 = (r_full >= r_start) & (r_full <= r0)
        r_segment_start_r0 = r_full[mask_start_r0]
        v_segment_start_r0 = v_full[mask_start_r0]
        idx_sort = np.argsort(r_segment_start_r0)
        r_segment_start_r0 = r_segment_start_r0[idx_sort]
        v_segment_start_r0 = v_segment_start_r0[idx_sort]
        travel_time_start_r0 = np.trapz(1.0 / v_segment_start_r0, x=r_segment_start_r0)  # seconds

        # Backward: r0 to r1
        mask_r0_r1 = (r_full >= r0) & (r_full <= r1)
        r_segment_r0_r1 = r_full[mask_r0_r1]
        v_segment_r0_r1 = v_full[mask_r0_r1]
        idx_sort = np.argsort(r_segment_r0_r1)
        r_segment_r0_r1 = r_segment_r0_r1[idx_sort]
        v_segment_r0_r1 = v_segment_r0_r1[idx_sort]
        travel_time_r0_r1 = np.trapz(1.0 / v_segment_r0_r1, x=r_segment_r0_r1)  # seconds

        # Forward: r1 to r2 (1 AU)
        mask_r1_r2 = (r_full >= r1) & (r_full <= r2)
        r_segment_r1_r2 = r_full[mask_r1_r2]
        v_segment_r1_r2 = v_full[mask_r1_r2]
        idx_sort = np.argsort(r_segment_r1_r2)
        r_segment_r1_r2 = r_segment_r1_r2[idx_sort]
        v_segment_r1_r2 = v_segment_r1_r2[idx_sort]
        travel_time_r1_r2 = np.trapz(1.0 / v_segment_r1_r2, x=r_segment_r1_r2)  # seconds

        # Forward: r2 to r3 (1.524 AU)
        mask_r2_r3 = (r_full >= r2) & (r_full <= r3)
        r_segment_r2_r3 = r_full[mask_r2_r3]
        v_segment_r2_r3 = v_full[mask_r2_r3]
        idx_sort = np.argsort(r_segment_r2_r3)
        r_segment_r2_r3 = r_segment_r2_r3[idx_sort]
        v_segment_r2_r3 = v_segment_r2_r3[idx_sort]
        travel_time_r2_r3 = np.trapz(1.0 / v_segment_r2_r3, x=r_segment_r2_r3)  # seconds

        # Determine dates
        fecha_llegada_r1 = initial_date
        dt_start_r0 = timedelta(seconds=travel_time_start_r0)
        dt_r0_r1 = timedelta(seconds=travel_time_r0_r1)
        fecha_salida_r_start = fecha_llegada_r1 - dt_r0_r1 - dt_start_r0
        fecha_departure_r0 = fecha_llegada_r1 - dt_r0_r1
        dt_r1_r2 = timedelta(seconds=travel_time_r1_r2)
        fecha_llegada_r2 = fecha_llegada_r1 + dt_r1_r2
        dt_r2_r3 = timedelta(seconds=travel_time_r2_r3)
        fecha_llegada_r3 = fecha_llegada_r2 + dt_r2_r3

        # 11.E) Domain Setup
        # ----------------------------------------------------------------------------------
        n_theta = 100
        n_phi = 100
        theta_vals = np.linspace(0, 2 * np.pi, n_theta)
        phi_vals = np.linspace(-np.pi/2, np.pi/2, n_phi)
        theta, phi = np.meshgrid(theta_vals, phi_vals)
        taper = np.cos(phi)

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        u, v = np.meshgrid(u, v)
        x_sun = R_s_km * np.sin(v) * np.cos(u)
        y_sun = R_s_km * np.sin(v) * np.sin(u)
        z_sun = R_s_km * np.cos(v)

        theta_planet = np.linspace(0, 2 * np.pi, 200)
        x_earth = a_earth * np.cos(theta_planet)
        y_earth = b_earth * np.sin(theta_planet)
        z_earth = np.zeros_like(theta_planet)
        x_mercury = a_mercury * np.cos(theta_planet)
        y_mercury = b_mercury * np.sin(theta_planet)
        z_mercury = np.zeros_like(theta_planet)
        x_venus = a_venus * np.cos(theta_planet)
        y_venus = b_venus * np.sin(theta_planet)
        z_venus = np.zeros_like(theta_planet)
        x_mars = a_mars * np.cos(theta_planet)
        y_mars = b_mars * np.sin(theta_planet)
        z_mars = np.zeros_like(theta_planet)

        range_au = 1.6 * AU_km  # Extend range to include Mars' orbit
        x_limits = (0, range_au)
        y_limits = (-range_au / 2, range_au / 2)
        z_limits = (-range_au / 2, range_au / 2)

        theta_sec = np.linspace(0, 2 * np.pi, 300)

        # 11.F) Animation Setup
        # ----------------------------------------------------------------------------------
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Use the total travel time from r_start to r2 (1 AU)
        total_travel_time = travel_time_start_r0 + travel_time_r0_r1 + travel_time_r1_r2
        time_per_frame = total_travel_time / num_frames  # Corrected line
        times = np.linspace(0, total_travel_time, num_frames)

        # Compute positions over the full trajectory
        r_positions = np.zeros(num_frames)
        r_positions[0] = r_start / 1000  # Start at 1.8 R_s in km
        for i in range(1, num_frames):
            mask_segment = (r_full_km >= r_positions[i-1]) & (r_full_km <= x_end)
            r_seg = r_full_km[mask_segment]
            v_seg = v_full_kms[mask_segment]
            idx_sort = np.argsort(r_seg)
            r_seg = r_seg[idx_sort]
            v_seg = v_seg[idx_sort]
            time_remaining = time_per_frame
            r_current = r_positions[i-1]
            j = np.searchsorted(r_seg, r_current)
            while time_remaining > 0 and j < len(r_seg) - 1:
                dr = r_seg[j+1] - r_seg[j]
                v_avg = (v_seg[j] + v_seg[j+1]) / 2
                dt = dr / v_avg
                if dt <= time_remaining:
                    r_current = r_seg[j+1]
                    time_remaining -= dt
                    j += 1
                else:
                    fraction = time_remaining / dt
                    r_current += fraction * dr
                    time_remaining = 0
            r_positions[i] = r_current

        # 11.G) Animation Computation with Progress Bar
        # ----------------------------------------------------------------------------------
        st.write("Generating animation for the CME...")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        time_per_frame_est = 0.1  # seconds per frame
        total_time_est = num_frames * time_per_frame_est
        start_time = datetime.now()

        def update(frame):
            progress = (frame + 1) / num_frames
            progress_bar.progress(progress)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if frame > 0:
                time_per_frame_est_actual = elapsed_time / frame
                remaining_time = time_per_frame_est_actual * (num_frames - frame - 1)
            else:
                remaining_time = total_time_est
            
            progress_text.text(f"Progress: {int(progress * 100)}% - Estimated remaining time: {remaining_time:.1f} seconds")

            ax.clear()
            r = r_positions[frame]
            scale = 1.0 if r == r_start / 1000 else (r / (r_start / 1000)) ** expansion_ratio
            R_current = R_0 * scale
            a_max_current = a_max_0 * scale
            b_max_current = b_max_0 * scale
            a_current = a_max_current * taper
            b_current = b_max_current * taper
            
            x = (R_current + a_current * np.cos(theta)) * np.cos(phi) + r
            y = (R_current + a_current * np.cos(theta)) * np.sin(phi)
            z = b_current * np.sin(theta)
            
            x_rot = x
            y_rot = y * np.cos(theta_x) - z * np.sin(theta_x)
            z_rot = y * np.sin(theta_x) + z * np.cos(theta_x)
            
            taper0 = np.cos(phi0)
            a0 = a_max_current * taper0
            b0 = b_max_current * taper0
            x_sec = (R_current + a0 * np.cos(theta_sec)) * np.cos(phi0) + r
            y_sec = (R_current + a0 * np.cos(theta_sec)) * np.sin(phi0)
            z_sec = b0 * np.sin(theta_sec)
            x_sec_rot = x_sec
            y_sec_rot = y_sec * np.cos(theta_x) - z_sec * np.sin(theta_x)
            z_sec_rot = y_sec * np.sin(theta_x) + z_sec * np.cos(theta_x)
            
            y_center = np.mean(y_sec_rot)
            z_center = np.mean(z_sec_rot)
            y_rot_trans = y_rot - y_center
            z_rot_trans = z_rot - z_center
            y_sec_rot_trans = y_sec_rot - y_center
            z_sec_rot_trans = z_sec_rot - z_center
            
            x_line = np.linspace(0, range_au, 50)
            y_line = np.zeros_like(x_line)
            z_line = np.zeros_like(x_line)
            
            y_proj = y_rot_trans.ravel()
            z_proj = z_rot_trans.ravel()
            alpha_angles = np.linspace(0, 2*np.pi, 180)
            y_contour = []
            z_contour = []
            for alpha_val in alpha_angles:
                dy = np.cos(alpha_val)
                dz = np.sin(alpha_val)
                dot_product = y_proj * dy + z_proj * dz
                idx_max = np.argmax(dot_product)
                y_contour.append(y_proj[idx_max])
                z_contour.append(z_proj[idx_max])
            y_contour = np.array(y_contour)
            z_contour = np.array(z_contour)
            ellipse_model = EllipseModel()
            if ellipse_model.estimate(np.vstack((y_contour, z_contour)).T):
                yc, zc, a_fit, b_fit, theta_fit = ellipse_model.params
                cme_size = 2 * a_fit
            else:
                cme_size = np.nan

            ax.plot_surface(x_rot, y_rot_trans, z_rot_trans, rstride=2, cstride=2, 
                            alpha=0.6, antialiased=True)
            ax.plot(
                x_sec_rot, y_sec_rot_trans, z_sec_rot_trans,
                color='red', linewidth=2,
                label=r'Section ($\phi_0$=' + f'{phi0_deg:.1f}°)')
            ax.plot(
                x_line, y_line, z_line,
                color='gray', linestyle='--', linewidth=1, alpha=0.5,
                label='X Axis')
            ax.plot_surface(x_sun, y_sun, z_sun, color='orange', alpha=0.8)
            ax.plot(
                x_earth, y_earth, z_earth,
                color='blue', linewidth=1,
                label='Earth Orbit')
            ax.plot(
                x_mercury, y_mercury, z_mercury,
                color='black', linewidth=0.5, alpha=0.5,
                label='Mercury Orbit')
            ax.plot(
                x_venus, y_venus, z_venus,
                color='black', linewidth=0.5, alpha=0.5,
                label='Venus Orbit')
            ax.plot(
                x_mars, y_mars, z_mars,
                color='red', linewidth=0.5, alpha=0.5,
                label='Mars Orbit')
            ax.scatter(
                satellite_position[0], satellite_position[1], satellite_position[2],
                color='black', s=20,
                label='Satellite')

            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.set_title(f'CME Expanding (X = {r/149597870.7:.2f} AU, Size = {cme_size/149597870.7:.2f} AU)')
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)
            ax.legend()
            return

        # 11.H) Animation Execution with Progress Bar
        # ----------------------------------------------------------------------------------
        with st.spinner("Generating animation..."):
            print("Generating CME simulation animation...")
            ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames), desc="Animation Progress"), 
                                        interval=1000//fps, blit=False)
            html_video = ani.to_html5_video()
            html_video = f'''
            <div style="display:flex; justify-content:center;">
            {html_video.replace('<video ', '<video style="max-width:100%; margin:auto; display:block;" ')}
            </div>
            '''
        st.components.v1.html(html_video, height=600)




        # ----------------------------------------------------------------------------------
        # 12) Solar Wind and Propagation along the Interplanetary Medium
        # ----------------------------------------------------------------------------------
        st.subheader("12) Velocity profile along the Interplanetary Medium")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_full / R_s, v_full / 1e3, label='CME Velocity v(r)', color='blue')
        ax.plot(r_full / R_s, v_sw(r_full) / 1e3, label='Solar Wind Speed v_sw(r)', color='red', linestyle='--')
        ax.axvline(x=r0 / R_s, color='green', linestyle='--', label='r0 (Departure at 2.5 R_s)')
        ax.axvline(x=r1 / R_s, color='purple', linestyle='--', label=f'r1 (Satellite at {distance:.2f} AU)')
        ax.axvline(x=r2 / R_s, color='orange', linestyle='--', label='r2 (Earth at 1 AU)')
        ax.axvline(x=r3 / R_s, color='red', linestyle='--', label='r3 (Mars at 1.52 AU)')
        ax.set_xlabel('Distance from the Sun (r/R_s)')
        ax.set_ylabel('Speed (km/s)')
        ax.set_xscale('log')
        ax.set_xlim([1.8, r_max / R_s])  # Start at 1.8 R_s
        ax.grid(True, which='both', ls='--')
        ax.legend()
        ax.set_title('Velocity Profile of the CME and Solar Wind')
        st.pyplot(fig)

        # ----------------------------------------------------------------------------------
        # 13) 3D Plot of the CME
        # ----------------------------------------------------------------------------------
        st.subheader("13) 3D Plot of the CME")

        # 13.A) CME Dimensions in AU for Plotting
        # ----------------------------------------------------------------------------------
        a_max_au = a_1 / AU_km  # [AU]
        b_max_au = b_1 / AU_km  # [AU]
        R_au = 4 * a_max_au     # [AU] Major radius of the torus

        # 13.B) Generation of the CME and Cut Section
        # ----------------------------------------------------------------------------------
        n_theta = 100
        n_phi = 100
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(-np.pi/2, np.pi/2, n_phi)
        theta, phi = np.meshgrid(theta, phi)

        taper = np.cos(phi)
        a = a_max_au * taper
        b = b_max_au * taper

        x = (R_au + a * np.cos(theta)) * np.cos(phi)
        y = (R_au + a * np.cos(theta)) * np.sin(phi)
        z = b * np.sin(theta)

        x_rot = x
        y_rot = y * np.cos(theta_x) - z * np.sin(theta_x)
        z_rot = y * np.sin(theta_x) + z * np.cos(theta_x)

        taper0 = np.cos(phi0)
        a0 = a_max_au * taper0
        b0 = b_max_au * taper0
        theta_sec = np.linspace(0, 2 * np.pi, 300)
        x_sec = (R_au + a0 * np.cos(theta_sec)) * np.cos(phi0)
        y_sec = (R_au + a0 * np.cos(theta_sec)) * np.sin(phi0)
        z_sec = b0 * np.sin(theta_sec)

        x_sec_rot = x_sec
        y_sec_rot = y_sec * np.cos(theta_x) - z_sec * np.sin(theta_x)
        z_sec_rot = y_sec * np.sin(theta_x) + z_sec * np.cos(theta_x)

        # Compute the center of the cut section (before translation)
        center_x = R_au * np.cos(phi0)
        center_y = R_au * np.sin(phi0)
        center_z = 0
        center_x_rot = center_x
        center_y_rot = center_y * np.cos(theta_x) - center_z * np.sin(theta_x)
        center_z_rot = center_y * np.sin(theta_x) + center_z * np.cos(theta_x)

        # 13.C) Plotting with Plotly (3D Plot)
        # ----------------------------------------------------------------------------------
        fig = go.Figure()

        # Apply translation to CME surface
        x_rot_trans = x_rot
        y_rot_trans = y_rot - center_y_rot
        z_rot_trans = z_rot - center_z_rot

        fig.add_trace(go.Surface(
            x=x_rot_trans,
            y=y_rot_trans,
            z=z_rot_trans,
            colorscale='Blues',
            opacity=0.6,
            showscale=False,
            name="CME"
        ))

        # Apply translation to cut section
        x_sec_rot_trans = x_sec_rot
        y_sec_rot_trans = y_sec_rot - center_y_rot
        z_sec_rot_trans = z_sec_rot - center_z_rot

        fig.add_trace(go.Scatter3d(
            x=x_sec_rot_trans,
            y=y_sec_rot_trans,
            z=z_sec_rot_trans,
            mode='lines',
            line=dict(color='red', width=4),
            name=f"Cut section at φ={phi0_deg:.1f}°"
        ))

        # Apply translation to central axis
        theta_axis = np.linspace(0, 2 * np.pi, 300)
        x_axis = R_au * np.cos(theta_axis)
        y_axis = R_au * np.sin(theta_axis)
        z_axis = np.zeros_like(theta_axis)
        x_axis_rot = x_axis
        y_axis_rot = y_axis * np.cos(theta_x) - z_axis * np.sin(theta_x)
        z_axis_rot = y_axis * np.sin(theta_x) + z_axis * np.cos(theta_x)

        x_axis_rot_trans = x_axis_rot
        y_axis_rot_trans = y_axis_rot - center_y_rot
        z_axis_rot_trans = z_axis_rot - center_z_rot

        fig.add_trace(go.Scatter3d(
            x=x_axis_rot_trans,
            y=y_axis_rot_trans,
            z=z_axis_rot_trans,
            mode='lines',
            line=dict(color='black', width=2),
            name='Central axis'
        ))

        # Apply translation to director vector and arrowhead
        center_x_rot_trans = center_x_rot
        center_y_rot_trans = center_y_rot - center_y_rot  # Becomes 0
        center_z_rot_trans = center_z_rot - center_z_rot  # Becomes 0

        normal_orig = np.array([-np.sin(phi0), np.cos(phi0), 0])
        normal_rot = np.array([
            normal_orig[0],
            normal_orig[1] * np.cos(theta_x),
            normal_orig[1] * np.sin(theta_x)
        ])
        if normal_rot[0] < 0:
            normal_rot = -normal_rot
        normal_rot = normal_rot / np.linalg.norm(normal_rot)
        vector_length = 1.5 * a0  # Length of the arrow set to a0
        normal_rot_scaled = normal_rot * vector_length

        fig.add_trace(go.Scatter3d(
            x=[center_x_rot_trans, center_x_rot_trans + normal_rot_scaled[0]],
            y=[center_y_rot_trans, center_y_rot_trans + normal_rot_scaled[1]],
            z=[center_z_rot_trans, center_z_rot_trans + normal_rot_scaled[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Director vector'
        ))

        cone_size = 0.1
        fig.add_trace(go.Cone(
            x=[center_x_rot_trans + normal_rot_scaled[0]],
            y=[center_y_rot_trans + normal_rot_scaled[1]],
            z=[center_z_rot_trans + normal_rot_scaled[2]],
            u=[normal_rot_scaled[0]],
            v=[normal_rot_scaled[1]],
            w=[normal_rot_scaled[2]],
            sizemode="scaled",
            sizeref=cone_size,
            colorscale='Greens',
            showscale=False,
            name='Arrowhead'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X [AU]",
                yaxis_title="Y [AU]",
                zaxis_title="Z [AU]",
                aspectmode='data',
            ),
            title=f"CME Model at r = {distance:.2f} AU (satellite's position)", #    φ₀={phi0_deg:.1f}°, θₓ={theta_x_deg:.1f}°",
            showlegend=True,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)


        # ----------------------------------------------------------------------------------
        # 14) Projection onto the YZ Plane and Ellipse Fitting with Horizontal Cut
        # ----------------------------------------------------------------------------------

        st.subheader("14) Projection onto the YZ Plane")

        # 14.1) Projection onto the YZ Plane and Ellipse Fitting with Horizontal Cut
        # ----------------------------------------------------------------------------------
        # Apply translation to the projection coordinates (same as Section 13)
        y_proj = (y_rot - center_y_rot).ravel()
        z_proj = (z_rot - center_z_rot).ravel()

        alpha_angles = np.linspace(0, 2 * np.pi, 360)
        y_contour = []
        z_contour = []

        for alpha in alpha_angles:
            dy = np.cos(alpha)
            dz = np.sin(alpha)
            dot_product = y_proj * dy + z_proj * dz
            idx_max = np.argmax(dot_product)
            y_contour.append(y_proj[idx_max])
            z_contour.append(z_proj[idx_max])

        y_contour = np.array(y_contour)
        z_contour = np.array(z_contour)

        points_contour = np.vstack((y_contour, z_contour)).T
        ellipse = EllipseModel()
        if ellipse.estimate(points_contour):
            yc, zc, a_fit, b_fit, theta_fit = ellipse.params
            area_ellipse = np.pi * a_fit * b_fit
        else:
            st.write("Could not fit an ellipse to the contour.")
            yc, zc, a_fit, b_fit, theta_fit = 0, 0, 0, 0, 0
            area_ellipse = 0

        t = np.linspace(0, 2 * np.pi, 100)
        y_ellipse = yc + a_fit * np.cos(t) * np.cos(theta_fit) - b_fit * np.sin(t) * np.sin(theta_fit)
        z_ellipse = zc + a_fit * np.cos(t) * np.sin(theta_fit) + b_fit * np.sin(t) * np.cos(theta_fit)

        # Create a single Plotly figure
        fig_proj = go.Figure()

        # Add direct projection (hide from legend)
        fig_proj.add_trace(go.Scatter(
            x=y_proj,
            y=z_proj,
            mode='markers',
            marker=dict(size=1, opacity=0.3),
            name='Direct projection',
            showlegend=False  # Remove from legend
        ))

        # Add projected contour (hide from legend)
        fig_proj.add_trace(go.Scatter(
            x=y_contour,
            y=z_contour,
            mode='lines',
            line=dict(color='green', width=2),
            name='Projected contour',
            showlegend=False  # Remove from legend
        ))

        # Add fitted ellipse (rename to "Contour of the Projection")
        fig_proj.add_trace(go.Scatter(
            x=y_ellipse,
            y=z_ellipse,
            mode='lines',
            line=dict(color='red', width=2),
            name='Contour of the Projection'  # Renamed from 'Fitted ellipse'
        ))

        # Calculate the length of the horizontal cut at z = 0
        A = a_fit * np.sin(theta_fit)
        B = b_fit * np.cos(theta_fit)
        C = -zc

        cut_length = None
        R = np.sqrt(A**2 + B**2)
        if abs(C / R) <= 1:
            # Solve A * cos(t) + B * sin(t) = C
            phi = np.arctan2(B, A)
            cos_val = C / R
            delta_t = np.arccos(cos_val)
            t1 = phi + delta_t
            t2 = phi - delta_t

            # Compute y-coordinates at these points
            y1 = yc + a_fit * np.cos(t1) * np.cos(theta_fit) - b_fit * np.sin(t1) * np.sin(theta_fit)
            y2 = yc + a_fit * np.cos(t2) * np.cos(theta_fit) - b_fit * np.sin(t2) * np.sin(theta_fit)

            # Length of the horizontal cut
            cut_length = abs(y1 - y2)

            # Add the horizontal cut to the plot (keep in legend)
            fig_proj.add_trace(go.Scatter(
                x=[y1, y2],
                y=[0, 0],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                name=f'Horizontal cut (Length: {cut_length:.3f} AU)'
            ))
        else:
            st.write("The line z = 0 does not intersect the ellipse (z = 0 is outside the ellipse's z-range).")
            z_min = min(z_ellipse)
            z_max = max(z_ellipse)
            st.write(f"Z-range of the ellipse: [{z_min:.3f}, {z_max:.3f}] AU")

        # Update the layout of the figure
        fig_proj.update_layout(
            title=f"Projection onto YZ Plane with Cut at z = 0 (Ecliptic Plane) at r = {distance:.2f} AU",
            xaxis_title="Y [AU]",
            yaxis_title="Z [AU]",
            showlegend=True,
            height=600,
            width=600,
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        # Display the figure
        st.plotly_chart(fig_proj, use_container_width=True)

        # Display the length of the horizontal cut and the projected area below the figure
        if cut_length is not None:
            st.markdown(f"- **Length of the horizontal cut at z = 0**: {cut_length:.3f} AU")
        else:
            st.markdown(f"- **Length of the horizontal cut at z = 0**: Not applicable (z = 0 does not intersect the ellipse)")
        st.markdown(f"- **Projected area**: {area_ellipse:.3f} AU²")



        # 14.B) Calculate Horizontal Cut Length at r_2 (1 AU)
        # ----------------------------------------------------------------------------------

        # Use variables from section 11
        r1_m = r1  # Distance at satellite position (meters)
        r2_m = r2  # Distance at 1 AU (meters)
        expansion_ratio_val = expansion_ratio  # From section 11.B

        # CME dimensions at r_1 (from section 13.A)
        a_max_au_r1 = a_1 / AU_km  # [AU]
        b_max_au_r1 = b_1 / AU_km  # [AU]
        R_au_r1 = 4 * a_max_au_r1  # [AU]

        # Scale dimensions to r_2
        scale_factor = (r2_m / r1_m) ** expansion_ratio_val
        a_max_au_r2 = a_max_au_r1 * scale_factor
        b_max_au_r2 = b_max_au_r1 * scale_factor
        R_au_r2 = R_au_r1 * scale_factor

        # Generate CME geometry at r_2 (similar to section 13.B)
        n_theta = 100
        n_phi = 100
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(-np.pi/2, np.pi/2, n_phi)
        theta, phi = np.meshgrid(theta, phi)

        taper = np.cos(phi)
        a_r2 = a_max_au_r2 * taper
        b_r2 = b_max_au_r2 * taper

        x_r2 = (R_au_r2 + a_r2 * np.cos(theta)) * np.cos(phi)
        y_r2 = (R_au_r2 + a_r2 * np.cos(theta)) * np.sin(phi)
        z_r2 = b_r2 * np.sin(theta)

        # Apply rotation (same as section 13.B)
        x_rot_r2 = x_r2
        y_rot_r2 = y_r2 * np.cos(theta_x) - z_r2 * np.sin(theta_x)
        z_rot_r2 = y_r2 * np.sin(theta_x) + z_r2 * np.cos(theta_x)

        # Project onto YZ plane (similar to section 13.D)
        y_proj_r2 = y_rot_r2.ravel()
        z_proj_r2 = z_rot_r2.ravel()

        alpha_angles = np.linspace(0, 2 * np.pi, 360)
        y_contour_r2 = []
        z_contour_r2 = []

        for alpha in alpha_angles:
            dy = np.cos(alpha)
            dz = np.sin(alpha)
            dot_product = y_proj_r2 * dy + z_proj_r2 * dz
            idx_max = np.argmax(dot_product)
            y_contour_r2.append(y_proj_r2[idx_max])
            z_contour_r2.append(z_proj_r2[idx_max])

        y_contour_r2 = np.array(y_contour_r2)
        z_contour_r2 = np.array(z_contour_r2)

        # Fit an ellipse to the projected contour
        points_contour_r2 = np.vstack((y_contour_r2, z_contour_r2)).T
        ellipse_r2 = EllipseModel()
        if ellipse_r2.estimate(points_contour_r2):
            yc_r2, zc_r2, a_fit_r2, b_fit_r2, theta_fit_r2 = ellipse_r2.params
            area_ellipse_r2 = np.pi * a_fit_r2 * b_fit_r2
        else:
            st.write("Could not fit an ellipse to the contour at r2.")
            yc_r2, zc_r2, a_fit_r2, b_fit_r2, theta_fit_r2 = 0, 0, 0, 0, 0
            area_ellipse_r2 = 0

        # Calculate the length of the horizontal cut at z = 0
        A_r2 = a_fit_r2 * np.sin(theta_fit_r2)
        B_r2 = b_fit_r2 * np.cos(theta_fit_r2)
        C_r2 = -zc_r2

        cut_length_r2 = None
        R_r2 = np.sqrt(A_r2**2 + B_r2**2)
        if abs(C_r2 / R_r2) <= 1:
            # Solve A * cos(t) + B * sin(t) = C
            phi_r2 = np.arctan2(B_r2, A_r2)
            cos_val_r2 = C_r2 / R_r2
            delta_t_r2 = np.arccos(cos_val_r2)
            t1_r2 = phi_r2 + delta_t_r2
            t2_r2 = phi_r2 - delta_t_r2

            # Compute y-coordinates at these points
            y1_r2 = yc_r2 + a_fit_r2 * np.cos(t1_r2) * np.cos(theta_fit_r2) - b_fit_r2 * np.sin(t1_r2) * np.sin(theta_fit_r2)
            y2_r2 = yc_r2 + a_fit_r2 * np.cos(t2_r2) * np.cos(theta_fit_r2) - b_fit_r2 * np.sin(t2_r2) * np.sin(theta_fit_r2)

            # Length of the horizontal cut
            cut_length_r2 = abs(y1_r2 - y2_r2)
        else:
            st.write("The line z = 0 does not intersect the ellipse at r_2 (z = 0 is outside the ellipse's z-range).")
            # Compute z-range for diagnostic
            t = np.linspace(0, 2 * np.pi, 100)
            z_ellipse_r2 = zc_r2 + a_fit_r2 * np.cos(t) * np.sin(theta_fit_r2) + b_fit_r2 * np.sin(t) * np.cos(theta_fit_r2)
            z_min_r2 = min(z_ellipse_r2)
            z_max_r2 = max(z_ellipse_r2)
            st.write(f"Z-range of the ellipse at r2: [{z_min_r2:.3f}, {z_max_r2:.3f}] AU")

        # Display the results
        st.markdown("**Results at 1AU**:")

        if cut_length_r2 is not None:
            st.markdown(f"- **Length of the horizontal cut at z = 0**: {cut_length_r2:.3f} AU")
        else:
            st.markdown(f"- **Length of the horizontal cut at z = 0**: Not applicable (z = 0 does not intersect the ellipse)")
        st.markdown(f"- **Projected area**: {area_ellipse_r2:.3f} AU²")



        # 15) Trajectory of the CME contained in the XY plane
        # ----------------------------------------------------------------------------------
        st.subheader("15) Evolution of the length of the CME contained in the XY plane")

        # From previous sections
        L_1 = cut_length  # From section 14.1
        L_2 = cut_length_r2  # From section 14.B

        # Convert r_1, r_2, and r_3 from meters to AU (r_3 from Section 11.A)
        AU_m = 149597870.7 * 1000  # meters per AU
        r1_au = r1_m / AU_m
        r2_au = r2_m / AU_m
        r3_au = r3 / AU_m  # r3 is 1.524 AU, defined in Section 11.A

        # Check if cut lengths are available
        if L_1 is not None and L_2 is not None:
            # Calculate gamma_1 = arctan(L_1 / (2 * r_1))
            gamma_1_rad = np.arctan(L_1 / (2 * r1_au))
            gamma_1_deg = np.degrees(gamma_1_rad)

            # Calculate gamma_2 = arctan(L_2 / (2 * r_2))
            gamma_2_rad = np.arctan(L_2 / (2 * r2_au))
            gamma_2_deg = np.degrees(gamma_2_rad)

            # Calculate gamma_3 = arctan(L_3 / (2 * r_3)) after computing L_3
            # (We’ll compute L_3 below using the power-law fit)



            # Fit a power-law function f(r) = A * r^k such that f(r_1) = L_1/2 and f(r_2) = L_2/2
            if L_1 > 0 and L_2 > 0 and r2_au > r1_au * 1.08:  # Ensure L_1 and L_2 are positive to avoid log issues
                # Solve for k
                k = np.log(L_2 / L_1) / np.log(r2_au / r1_au)
                # Solve for A
                A = (L_1 / 2) / (r1_au ** k)
            else:
                k = 1.4
                A = (L_1 / 2) / (r1_au ** k)

            # Define the function f(r) = A * r^k
            def f(r):
                return A * (r ** k)

            # Compute L_3 at r_3 using the power-law fit
            L_3_half = f(r3_au)  # L_3/2
            L_3 = 2 * L_3_half
            gamma_3_rad = np.arctan(L_3 / (2 * r3_au))
            gamma_3_deg = np.degrees(gamma_3_rad)


            # Generate points for plotting
            r_values = np.linspace(0.1, 1.6, 100)  # From 0.1 AU to 1.6 AU
            f_values = f(r_values)
            f_neg_values = -f_values  # Negative for symmetry

            # Crea la figura
            fig = go.Figure()

            # 1) Power‐law positive (blue, with legend)
            fig.add_trace(go.Scatter(
                x=r_values,
                y=f_values,
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'f(r) = {A:.3f}·r^{k:.2f}'
            ))

            # 2) Power‐law negative (blue, without legend)
            fig.add_trace(go.Scatter(
                x=r_values,
                y=-f_values,
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))


            # 3) Vertical lines contained between the curves, with English legend labels
            f1 = f(r1_au)
            f2 = f(r2_au)
            f3 = f(r3_au)

            # Line at r1 (Satellite)
            fig.add_trace(go.Scatter(
                x=[r1_au, r1_au],
                y=[-f1, f1],
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name=f'r1 = {r1_au:.2f} AU (Satellite)'
            ))

            # Line at r2 (Earth)
            fig.add_trace(go.Scatter(
                x=[r2_au, r2_au],
                y=[-f2, f2],
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name=f'r2 = {r2_au:.2f} AU (Earth)'
            ))

            # Line at r3 (Mars)
            fig.add_trace(go.Scatter(
                x=[r3_au, r3_au],
                y=[-f3, f3],
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name=f'r3 = {r3_au:.2f} AU (Mars)'
            ))


            # 4) Layout actualizado
            fig.update_layout(
                title="Contained Trajectory of the CME in the XY Plane",
                xaxis_title="x [AU]",
                yaxis_title="z [AU]",
                showlegend=True,
                height=500,
                width=700,
                xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
                yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1)
            )

            # 5) Muestra el plot
            st.plotly_chart(fig, use_container_width=True)

            # Recompute the angles as γ_i = arctan( f(r_i) / r_i )
            gamma_1_rad = np.arctan(f1 / r1_au)
            gamma_2_rad = np.arctan(f2 / r2_au)
            gamma_3_rad = np.arctan(f3 / r3_au)

            gamma_1_deg = np.degrees(gamma_1_rad)
            gamma_2_deg = np.degrees(gamma_2_rad)
            gamma_3_deg = np.degrees(gamma_3_rad)

            # 6) Mostrar los tres ángulos debajo del gráfico
            st.markdown("#### Angular widths")
            st.latex(r"\gamma_1 = \arctan\!\bigl(\tfrac{f(r_1)}{r_1}\bigr) =\pm %.2f^\circ" % gamma_1_deg)
            st.latex(r"\gamma_2 = \arctan\!\bigl(\tfrac{f(r_2)}{r_2}\bigr) =\pm %.2f^\circ" % gamma_2_deg)
            st.latex(r"\gamma_3 = \arctan\!\bigl(\tfrac{f(r_3)}{r_3}\bigr) =\pm %.2f^\circ" % gamma_3_deg)

        else:
            st.markdown("#### Angular Widths:")
            st.markdown("- **γ₁, γ₂, and γ₃**: Cannot be calculated due to missing cut length(s).")
            st.markdown("#### Fitted Power-Law Functions:")
            st.markdown("- Cannot fit power-law functions due to missing cut length(s).")
                    



        # 15.B) Proporciones del segmento en z=0 para y < 0 y y > 0
        # ----------------------------------------------------------------------------------
        # Usar los parámetros de la elipse ajustada de la sección 13.D
        yc, zc, a, b, theta = ellipse.params  # Asume que ellipse.params está disponible

        # Calcular los puntos de intersección en z=0
        A = a * np.sin(theta)
        B = b * np.cos(theta)
        C = -zc
        R = np.sqrt(A**2 + B**2)

        if abs(C / R) <= 1:
            # Calcular t1 y t2
            phi = np.arctan2(B, A)
            t1 = phi + np.arccos(C / R)
            t2 = phi - np.arccos(C / R)
            
            # Calcular coordenadas y
            y1 = yc + a * np.cos(t1) * np.cos(theta) - b * np.sin(t1) * np.sin(theta)
            y2 = yc + a * np.cos(t2) * np.cos(theta) - b * np.sin(t2) * np.sin(theta)
            
            # Longitud total
            L = abs(y1 - y2)
            
            # Determinar proporciones
            if y1 > 0 and y2 > 0:
                prop_y_pos = 1.0
                prop_y_neg = 0.0
            elif y1 < 0 and y2 < 0:
                prop_y_pos = 0.0
                prop_y_neg = 1.0
            else:
                # Un punto en y > 0, otro en y < 0
                prop_y_pos = abs(y1) / L if y1 > 0 else abs(y2) / L
                prop_y_neg = abs(y2) / L if y2 < 0 else abs(y1) / L
            
            DeltaL = (prop_y_neg - prop_y_pos)/2
            # Calcular el ángulo de rotación
            theta_rot = np.arctan(DeltaL / (r1_au))
            theta_rot_deg = np.degrees(np.arctan(DeltaL / (r1_au)))
            
            st.markdown("#### Segment Proportions at z=0:")
            st.markdown(f"- Total longitude contained in XY at r1: {L:.3f} AU")
            st.markdown(f"- Portion at the right of the satellite: {prop_y_pos:.3f} ({prop_y_pos*100:.1f}%)")
            st.markdown(f"- Portion at the left: {prop_y_neg:.3f} ({prop_y_neg*100:.1f}%)")
            st.markdown(f"- Ecliptic Longitude difference between the CME's center and the Satellite: {np.degrees(theta_rot):.2f}°")
        else:
            st.markdown("#### Segment Proportions at z=0:")
            st.markdown("- La elipse no cruza z=0. Longitud = 0, proporciones indefinidas.")



        # ----------------------------------------------------------------------------------
        # 16) Summary of the Results
        # ----------------------------------------------------------------------------------
        st.subheader("Summary of the Results")

        # Total travel time from r0 (2.5 R_s) to r1
        travel_time_r0_to_r1 = travel_time_r0_r1
        days_r0_r1 = int(travel_time_r0_to_r1 // (24 * 3600))
        hours_r0_r1 = int((travel_time_r0_to_r1 % (24 * 3600)) // 3600)
        minutes_r0_r1 = int((travel_time_r0_to_r1 % 3600) // 60)

        # Total travel time from r0 (2.5 R_s) to r2 (1 AU)
        travel_time_r0_to_r2 = travel_time_r0_r1 + travel_time_r1_r2
        days_r0_r2 = int(travel_time_r0_to_r2 // (24 * 3600))
        hours_r0_r2 = int((travel_time_r0_to_r2 % (24 * 3600)) // 3600)
        minutes_r0_r2 = int((travel_time_r0_to_r2 % 3600) // 60)

        # Travel time from r2 to r3 (1.524 AU)
        days_r2_r3 = int(travel_time_r2_r3 // (24 * 3600))
        hours_r2_r3 = int((travel_time_r2_r3 % (24 * 3600)) // 3600)
        minutes_r2_r3 = int((travel_time_r2_r3 % 3600) // 60)

        # Total travel time from r0 (2.5 R_s) to r3 (1.524 AU)
        travel_time_r0_to_r3 = travel_time_r0_r1 + travel_time_r1_r2 + travel_time_r2_r3
        days_r0_r3 = int(travel_time_r0_to_r3 // (24 * 3600))
        hours_r0_r3 = int((travel_time_r0_to_r3 % (24 * 3600)) // 3600)
        minutes_r0_r3 = int((travel_time_r0_to_r3 % 3600) // 60)

        # Travel time from r1 to r2 (1 AU)
        days_r1_r2 = int(travel_time_r1_r2 // (24 * 3600))
        hours_r1_r2 = int((travel_time_r1_r2 % (24 * 3600)) // 3600)
        minutes_r1_r2 = int((travel_time_r1_r2 % 3600) // 60)

        # Average propagation speed from r0 (2.5 R_s) to r3 (1.524 AU)
        total_distance_r0_to_r3 = r3 - r0  # Distance from r0 to r3 in meters
        avg_speed = total_distance_r0_to_r3 / travel_time_r0_to_r3  # m/s
        avg_speed_kms = avg_speed / 1000  # km/s

        # Average propagation speed from r0 (Sun) to r2 (Earth)
        total_distance_r0_to_r2 = r2 - r0  # Distance from r0 to r3 in meters
        avg_speed_to_r2 = total_distance_r0_to_r2 / travel_time_r0_to_r2  # m/s
        avg_speed_kms_r0_r2 = avg_speed / 1000  # km/s

        # Convert r0 and r3 to appropriate units for display
        r0_in_rs = r0 / R_s  # r0 in units of R_s
        r3_in_au = r3 / AU   # r3 in units of AU

        # Summary with subsections and smaller subtitles
        st.markdown(f"""
        <h3 style='font-size:16px;'>Key Timestamps</h3>
        <ul>
            <li><b>Departure from r0 (2.5 R_s):</b> {fecha_departure_r0.strftime('%Y-%m-%d')}          <b>Time:</b> {fecha_departure_r0.strftime('%H:%M:%S')}</li>
            <li><b>Detected by the satellite on:</b> {fecha_llegada_r1.strftime('%Y-%m-%d')}          <b>Time:</b> {fecha_llegada_r1.strftime('%H:%M:%S')}</li>
            <li><b>Crosses the Earth's orbit on:</b> {fecha_llegada_r2.strftime('%Y-%m-%d')}          <b>Time:</b> {fecha_llegada_r2.strftime('%H:%M:%S')}</li>
            <li><b>Crosses Mars' orbit on:</b> {fecha_llegada_r3.strftime('%Y-%m-%d')}          <b>Time:</b> {fecha_llegada_r3.strftime('%H:%M:%S')}</li>
        </ul>

        <h3 style='font-size:16px;'>Propagation Times</h3>
        <ul>
            <li><b>Travel Time (Sun to Satellite):</b> {days_r0_r1} days, {hours_r0_r1} hours, {minutes_r0_r1} minutes</li>
            <li><b>Travel Time (Satellite to Earth):</b> {days_r1_r2} days, {hours_r1_r2} hours, {minutes_r1_r2} minutes</li>
            <li><b>Travel Time (Earth to Mars):</b> {days_r2_r3} days, {hours_r2_r3} hours, {minutes_r2_r3} minutes</li>
            <li><b>Travel Time (Sun to Earth):</b> {days_r0_r2} days, {hours_r0_r2} hours, {minutes_r0_r2} minutes</li>
            <li><b>Travel Time (Sun to Mars):</b> {days_r0_r3} days, {hours_r0_r3} hours, {minutes_r0_r3} minutes</li>
        </ul>

        <h3 style='font-size:16px;'>CME Properties</h3>
        <ul>
            <li><b>Total Volume of the CME:</b> {Vol1_AU:.2e} AU³</li>
            <li><b>Total Mass of the CME:</b> {m_cme:.2e} kg</li>
            <li><b>Average Propagation Speed ({r0_in_rs:.1f} R_s to 1 AU):</b> {avg_speed_kms_r0_r2:.2f} km/s</li>
            <li><b>Inclination of the CME:</b> {theta_x_deg:.2f}°</li>
            <li><b>Cut section angle:</b> {phi0_deg:.2f}°</li>
        </ul>
        """, unsafe_allow_html=True)


        # ----------------------------------------------------------------------------------
        # Section 17: Checking if the CME passes through Earth or Mars
        # ----------------------------------------------------------------------------------
        st.subheader("CME Interaction with Earth and Mars")

        # Parse the dates of interest
        earth_date = parse_time(fecha_llegada_r2)
        mars_date = parse_time(fecha_llegada_r3)

        # Calculate Earth's position at fecha_llegada_r2
        earth_coord = get_body_heliographic_stonyhurst('Earth', earth_date)
        earth_distance, earth_lon = get_position_data('Earth', earth_coord, earth_date)
        earth_lon = earth_lon % 360  # Ensure longitude is in [0, 360)

        # Calculate Mars' position at fecha_llegada_r3
        mars_coord = get_body_heliographic_stonyhurst('Mars', mars_date)
        mars_distance, mars_lon = get_position_data('Mars', mars_coord, mars_date)
        mars_lon = mars_lon % 360  # Ensure longitude is in [0, 360)

        # Compute the CME's center ecliptic longitude
        cme_lon = lon_ecliptic % 360  # Satellite's ecliptic longitude
        cme_center_lon = (cme_lon + np.degrees(theta_rot)) % 360  # Add the offset theta_rot (converted to degrees)

        # Compute the intervals of interaction
        # At Earth's distance (r_2), using gamma_2_deg
        earth_min_angle = (cme_center_lon - gamma_2_deg) % 360
        earth_max_angle = (cme_center_lon + gamma_2_deg) % 360

        # At Mars' distance (r_3), using gamma_3_deg
        mars_min_angle = (cme_center_lon - gamma_3_deg) % 360
        mars_max_angle = (cme_center_lon + gamma_3_deg) % 360

        # Display the calculated positions
        st.markdown(f"""
        **Earth's Position at {earth_date.strftime('%Y-%m-%d %H:%M:%S')}**:
        - Radial Distance: {earth_distance:.3f} AU
        - Ecliptic Longitude: {earth_lon:.2f}°

        **Mars' Position at {mars_date.strftime('%Y-%m-%d %H:%M:%S')}**:
        - Radial Distance: {mars_distance:.3f} AU
        - Ecliptic Longitude: {mars_lon:.2f}°
        """)

        # Display CME propagation details
        st.markdown("**CME Propagation:**")
        st.markdown(f"- Ecliptic Longitude of the center: {cme_center_lon:.2f}°")
        st.markdown(f"- Interval of interaction at Earth's Radial Distance: ({earth_min_angle:.2f}°, {earth_max_angle:.2f}°)")
        st.markdown(f"- Interval of interaction at Mars' Position: ({mars_min_angle:.2f}°, {mars_max_angle:.2f}°)")

        # Function to check if a longitude is within an interval, handling the 0°/360° boundary
        def is_within_interval(lon, min_angle, max_angle):
            # Normalize the longitude to [0, 360)
            lon = lon % 360
            min_angle = min_angle % 360
            max_angle = max_angle % 360
            
            # If the interval does not cross the 0°/360° boundary
            if min_angle <= max_angle:
                return min_angle <= lon <= max_angle
            else:
                # Interval crosses the 0°/360° boundary
                return lon >= min_angle or lon <= max_angle

        # Check if Earth's ecliptic longitude is within the interval at r_2
        earth_within_interval = is_within_interval(earth_lon, earth_min_angle, earth_max_angle)

        # Check if Mars' ecliptic longitude is within the interval at r_3
        mars_within_interval = is_within_interval(mars_lon, mars_min_angle, mars_max_angle)

        # Display the encounter analysis based on ecliptic longitude intervals
        st.markdown("**CME Encounter Analysis (Based on Ecliptic Longitude):**")
        if earth_within_interval:
            st.markdown(f"- **Earth**: The CME is likely to pass through Earth's position (Ecliptic Longitude {earth_lon:.2f}° is within [{earth_min_angle:.2f}°, {earth_max_angle:.2f}°]).")
        else:
            st.markdown(f"- **Earth**: The CME is unlikely to pass through Earth's position (Ecliptic Longitude {earth_lon:.2f}° is not within [{earth_min_angle:.2f}°, {earth_max_angle:.2f}°]).")
        if mars_within_interval:
            st.markdown(f"- **Mars**: The CME is likely to pass through Mars' position (Ecliptic Longitude {mars_lon:.2f}° is within [{mars_min_angle:.2f}°, {mars_max_angle:.2f}°]).")
        else:
            st.markdown(f"- **Mars**: The CME is unlikely to pass through Mars' position (Ecliptic Longitude {mars_lon:.2f}° is not within [{mars_min_angle:.2f}°, {mars_max_angle:.2f}°]).")

        # # Additional note on limitations
        # st.markdown("*Note*: This analysis is based solely on ecliptic longitude intervals and does not account for radial distance differences or the CME's 3D trajectory and temporal evolution.")


        # ----------------------------------------------------------------------------------
        # 18) Plot of the Interplanetary Scene
        # ----------------------------------------------------------------------------------


    if best_combination is None:
        return None

    return (best_combination, B_components_fit, trajectory_vectors,
            viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt)
