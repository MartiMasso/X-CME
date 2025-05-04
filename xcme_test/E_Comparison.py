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
import pandas as pd


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
    return r, alpha

# Function to compute the fitting (with cache)
st.cache_data.clear()
@st.cache_data
def models_Comparison(data, initial_point, final_point, initial_date, final_date,  distance, lon_ecliptic):
    ddoy_data = data['ddoy'].values
    B_data = data['B'].values
    Bx_data = data['Bx'].values
    By_data = data['By'].values
    Bz_data = data['Bz'].values
    Vsw_data = data['Vsw'].values
    data_tuple = (ddoy_data, B_data, Bx_data, By_data, Bz_data)

    initial_point = max(1, initial_point)
    Npt = final_point - initial_point
    Npt_resolution = 400

    B_exp = B_data[initial_point - 1:final_point - 1]
    Bx_exp = Bx_data[initial_point - 1:final_point - 1]
    By_exp = By_data[initial_point - 1:final_point - 1]
    Bz_exp = Bz_data[initial_point - 1:final_point - 1]
    ddoy_exp = ddoy_data[initial_point - 1:final_point - 1]
    Vsw_exp = Vsw_data[initial_point - 1:final_point - 1]

    Vsw_avg = np.mean(Vsw_exp)
    time_span_fr = (ddoy_exp[-1] - ddoy_exp[0]) * 24 * 3600
    length_L1 = np.abs(Vsw_avg * time_span_fr)

    ### Test L1 data ### 
    days = 2
    ts = days * 3600 * 24
    Vsw = 430
    length_L1 = ts*Vsw

    def project_to_plane(point, normal, d):
        dot_product = np.dot(normal, point)
        t = (d - dot_product) / np.dot(normal, normal)
        projected_point = point + t * normal
        return projected_point


    # Test Combination 1 (quite long)
    z0_range = np.arange(-0.6, 0.7 + 0.1, 0.15)
    angle_x_range = np.arange(0.1, np.pi - 0.1, 0.4)
    angle_y_range = np.arange(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 0.4)
    angle_z_range = -np.arange(0, np.pi, 0.4)
    delta_range = np.arange(0.6, 1 + 0.1, 0.2)

    # # Test Combination 1 (quite short)
    # z0_range = np.arange(0.2, 0.7 + 0.1, 0.1)
    # angle_x_range = np.arange(0.1, np.pi - 0.1, 0.3)
    # # angle_y_range = np.arange(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 0.9)
    # angle_y_range =  np.array([np.radians(0), np.radians(10), np.radians(20)])
    # angle_z_range = np.arange(0, np.pi, 0.3)
    # delta_range = np.arange(0.6, 1 + 0.1, 0.1)

    # # Test Combination 2
    # z0_range = np.arange(-0.5, 0.5 + 0.1, 0.2)
    # angle_x_range = np.arange(0.1, np.pi - 0.1, 0.4)
    # angle_y_range = np.arange(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 0.5)
    # angle_z_range = np.arange(0, np.pi, 0.5)
    # delta_range = np.array([0.7, 0.8])


    # # Synthetic FR combination
    # z0_range = np.array([0.7, 0.8, 0.9])
    # angle_x_range = np.array([np.radians(60)])
    # angle_y_range = np.array([np.radians(0)])
    # angle_z_range = np.array([-np.radians(30)])
    # delta_range = np.array([0.7, 0.8, 0.9])

    total_iterations = len(z0_range) * len(angle_x_range) * len(angle_y_range) * len(angle_z_range) * len(delta_range)
    st.write("Total Iterations:", total_iterations)

    best_R2_general = -np.inf
    best_R2_radial = -np.inf
    best_R2_oscillatory = -np.inf

    best_combination = None
    B_components_fit = None
    trajectory_vectors = None
    viz_3d_vars_opt = None
    viz_2d_local_vars_opt = None
    viz_2d_rotated_vars_opt = None

    progress_bar = st.progress(0)
    progress_text = st.empty()
    current_iteration = 0

    # ----------------------------------------------------------------------------------
    # Main Loop
    # ----------------------------------------------------------------------------------
    for z0 in z0_range:
        for angle_x in angle_x_range:
            for angle_y in angle_y_range:
                for angle_z in angle_z_range:
                    for delta in delta_range:
                        current_iteration += 1
                        progress_percent = int((current_iteration / total_iterations) * 100)
                        progress_bar.progress(progress_percent)
                        progress_text.text(f"Processing... {progress_percent}% completed")

                        # ----------------------------------------------------------------------------------
                        # 1. Rotated Cylinder Computation
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
                        R_T = rotation_matrix_inv

                        # 1.2 Cylinder Coordinates Centered at Origin
                        # ----------------------------------------------------------------------------------
                        theta = np.linspace(0, 2 * np.pi, N)
                        z = np.linspace(-1.5 * a, 1.5 * a, N)
                        Theta, Z = np.meshgrid(theta, z)
                        X = a * np.cos(Theta)
                        Y = b * np.sin(Theta)
                        Z = Z  # Already centered in Z

                        # 1.3 Rotate Cylinder
                        # ----------------------------------------------------------------------------------
                        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
                        rotated_points = np.dot(rotation_matrix, np.vstack([X_flat, Y_flat, Z_flat]))
                        X_rot, Y_rot, Z_rot = rotated_points[0].reshape(N, N), rotated_points[1].reshape(N, N), rotated_points[2].reshape(N, N)

                        # ----------------------------------------------------------------------------------
                        # 2. Cylinder Intersection with Plane y = 0
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
                        X_ellipse = xc + a_ellipse * np.cos(t) * np.cos(theta) - b_ellipse * np.sin(t) * np.sin(theta)
                        Z_ellipse = zc + a_ellipse * np.cos(t) * np.sin(theta) + b_ellipse * np.sin(t) * np.cos(theta)

                        # 2.4 Center the Cylinder and Ellipse at Origin
                        # ----------------------------------------------------------------------------------
                        X_rot_centered = X_rot - xc
                        Y_rot_centered = Y_rot  # Y remains unchanged for y=0 plane
                        Z_rot_centered = Z_rot - zc
                        X_ellipse_centered = X_ellipse - xc
                        Z_ellipse_centered = Z_ellipse - zc

                        # ----------------------------------------------------------------------------------
                        # 3. Vertical Cut in the Section Ellipse
                        # ----------------------------------------------------------------------------------
                        Z_max = np.max(Z_ellipse_centered)
                        Z_min = np.min(Z_ellipse_centered)
                        z_cut = z0 * Z_max

                        # 3.1 Solve the System
                        # ----------------------------------------------------------------------------------
                        A_quad = (np.cos(theta)**2) / a_ellipse**2 + (np.sin(theta)**2) / b_ellipse**2
                        B_quad = 2 * ((np.cos(theta) * np.sin(theta)) / a_ellipse**2 - (np.cos(theta) * np.sin(theta)) / b_ellipse**2) * z_cut
                        C_quad = ((np.sin(theta)**2) / a_ellipse**2 + (np.cos(theta)**2) / b_ellipse**2) * z_cut**2 - 1

                        discriminant = B_quad**2 - 4 * A_quad * C_quad

                        if discriminant >= 0:
                            x1 = (-B_quad + np.sqrt(discriminant)) / (2 * A_quad)
                            x2 = (-B_quad - np.sqrt(discriminant)) / (2 * A_quad)
                            X_intersections = np.array([x1, x2])
                            Z_intersections = np.full_like(X_intersections, z_cut)

                            actual_length = np.abs(x2 - x1)
                            scale_factor = length_L1 / actual_length

                            # 3.2 Scale Coordinates
                            # ----------------------------------------------------------------------------------
                            X_ellipse_scaled = scale_factor * X_ellipse_centered
                            Z_ellipse_scaled = scale_factor * Z_ellipse_centered
                            x1_scaled = scale_factor * x1
                            x2_scaled = scale_factor * x2
                            Z_max_scaled = Z_max * scale_factor
                            z_cut_scaled = z_cut * scale_factor
                            X_intersections_scaled = np.array([x1_scaled, x2_scaled])
                            Z_intersections_scaled = np.full_like(X_intersections_scaled, z_cut_scaled)
                            X_rot_scaled = scale_factor * X_rot_centered
                            Y_rot_scaled = scale_factor * Y_rot_centered
                            Z_rot_scaled = scale_factor * Z_rot_centered

                            # 3.3 Satellite Trajectory
                            # ----------------------------------------------------------------------------------
                            xs = np.linspace(x1_scaled, x2_scaled, num=100)
                            ys = np.zeros_like(xs)
                            zs = np.full_like(xs, z_cut_scaled)

                            # 3.4 Ellipse Boundaries and Reference Lines
                            # ----------------------------------------------------------------------------------
                            x_min = np.min(X_ellipse_scaled)
                            x_max = np.max(X_ellipse_scaled)
                            z_min = np.min(Z_ellipse_scaled)
                            z_max = np.max(Z_ellipse_scaled)
                            margin = 0.1 * (x_max - x_min)
                            x_limits = [x_min - margin, x_max + margin]
                            z_limits = [z_min - margin, z_max + margin]

                            max_z_index = np.argmax(Z_ellipse_scaled)
                            x_max_z = X_ellipse_scaled[max_z_index]

                            indices_z_zero = np.where(np.abs(Z_ellipse_scaled - 0) < 1e-2)[0]
                            x_z_zero_limits = np.sort(X_ellipse_scaled[indices_z_zero]) if len(indices_z_zero) >= 2 else [x_min, x_max]
                            indices_x_max_z = np.where(np.abs(X_ellipse_scaled - x_max_z) < 1e-2)[0]
                            z_x_max_z_limits = np.sort(Z_ellipse_scaled[indices_x_max_z]) if len(indices_x_max_z) >= 2 else [z_min, z_max]

                            z_mid = (z_x_max_z_limits[0] + z_x_max_z_limits[1]) / 2
                            z_upper_limit = z_x_max_z_limits[1]
                            z_lower_limit_upper_half = z_mid
                            height_upper_half = z_upper_limit - z_lower_limit_upper_half
                            relative_position = z_cut_scaled - z_lower_limit_upper_half
                            percentage_in_upper_half = (relative_position / height_upper_half) * 100 if height_upper_half > 0 else 0

                            # # 3.6 Scaled Ellipse Parameters Output
                            # # ----------------------------------------------------------------------------------
                            # st.write("--- Scaled Ellipse Parameters ---")
                            # st.write(f"Experimental length L1: {length_L1:.6f}")
                            # st.write(f"Applied scale factor: {scale_factor:.6f}")
                            # st.write(f"New distance between intersection points: {np.abs(x2_scaled - x1_scaled):.6f}")
                            # st.write(f"New scaled intersection points: ({x1_scaled:.6f}, {z_cut_scaled:.6f}) and ({x2_scaled:.6f}, {z_cut_scaled:.6f})")
                            # st.write(f"Percentage of the trajectory height in the upper half of maximum z: {percentage_in_upper_half:.2f}%")

                        else:
                            st.write(f"\nNo real solutions found for the cut at Z = {z_cut}")


                        # ----------------------------------------------------------------------------------
                        # 4. Projection onto the Transverse Plane
                        # ----------------------------------------------------------------------------------
                       
                        # 4.1 Compute Cylinder Axis and Plane Equation
                        # ----------------------------------------------------------------------------------
                        axis_cylinder = rotation_matrix @ np.array([0, 0, 1])
                        axis_cylinder_norm = axis_cylinder / np.linalg.norm(axis_cylinder)
                        center_point = np.array([xc, 0, zc])
                        d = np.dot(axis_cylinder, center_point)

                        # 4.2 Projection Function
                        # ----------------------------------------------------------------------------------
                        def project_to_plane(point, normal, d):
                            dot_product = np.dot(normal, point)
                            t = (d - dot_product) / np.dot(normal, normal)
                            projected_point = point + t * normal
                            return projected_point

                        # 4.3 Project Ellipse, Intersections, and Trajectory
                        # ----------------------------------------------------------------------------------
                        X_ellipse_3d = np.vstack([X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled]).T
                        projected_ellipse = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in X_ellipse_3d])
                        X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse = projected_ellipse[:, 0], projected_ellipse[:, 1], projected_ellipse[:, 2]

                        intersection_points = np.array([[x1_scaled, 0, z_cut_scaled], [x2_scaled, 0, z_cut_scaled]])
                        projected_intersections = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in intersection_points])
                        X_proj_inter, Y_proj_inter, Z_proj_inter = projected_intersections[:, 0], projected_intersections[:, 1], projected_intersections[:, 2]

                        n_points = len(Bx_exp)  # Assuming Bx_exp is defined elsewhere
                        X_traj = np.linspace(x1_scaled, x2_scaled, n_points)
                        Y_traj = np.zeros_like(X_traj)
                        Z_traj = np.full_like(X_traj, z_cut_scaled)
                        trajectory_points = np.vstack([X_traj, Y_traj, Z_traj]).T
                        projected_trajectory = np.array([project_to_plane(p, axis_cylinder_norm, d) for p in trajectory_points])
                        X_proj_traj, Y_proj_traj, Z_proj_traj = projected_trajectory[:, 0], projected_trajectory[:, 1], projected_trajectory[:, 2]

                        # st.write("--- Projected Intersection Points ---")
                        # st.write(f"Point 1: ({X_proj_inter[0]:.6f}, {Y_proj_inter[0]:.6f}, {Z_proj_inter[0]:.6f})")
                        # st.write(f"Point 2: ({X_proj_inter[1]:.6f}, {Y_proj_inter[1]:.6f}, {Z_proj_inter[1]:.6f})")

                        # 4.4 Ellipse Fitting and Rotation
                        # ----------------------------------------------------------------------------------
                        ellipse_model = EllipseModel()
                        ellipse_points = np.column_stack((X_proj_ellipse, Y_proj_ellipse))
                        ellipse_model.estimate(ellipse_points)
                        xc_proj, yc_proj, a_proj, b_proj, theta_proj = ellipse_model.params

                        center_x_proj = np.mean(X_proj_ellipse)
                        center_y_proj = np.mean(Y_proj_ellipse)

                        slope_traj = (Y_proj_traj[-1] - Y_proj_traj[0]) / (X_proj_traj[-1] - X_proj_traj[0]) if (X_proj_traj[-1] - X_proj_traj[0]) != 0 else float('inf')
                        trajectory_angle = np.arctan2(Y_proj_traj[-1] - Y_proj_traj[0], X_proj_traj[-1] - X_proj_traj[0])

                        rotation_align = np.array([[np.cos(-trajectory_angle), -np.sin(-trajectory_angle)], [np.sin(-trajectory_angle), np.cos(-trajectory_angle)]])
                        rotation_180 = np.array([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]])
                        rotation_combined = rotation_180 @ rotation_align

                        ellipse_2d = np.vstack([X_proj_ellipse, Y_proj_ellipse]).T
                        ellipse_rotated = (rotation_combined @ (ellipse_2d - np.array([center_x_proj, center_y_proj]).T).T).T + np.array([center_x_proj, center_y_proj])
                        X_ellipse_rotated, Y_ellipse_rotated = ellipse_rotated[:, 0], ellipse_rotated[:, 1]

                        trajectory_2d = np.vstack([X_proj_traj, Y_proj_traj]).T
                        trajectory_rotated = (rotation_combined @ (trajectory_2d - np.array([center_x_proj, center_y_proj]).T).T).T + np.array([center_x_proj, center_y_proj])
                        X_traj_rotated, Y_traj_rotated = trajectory_rotated[:, 0], trajectory_rotated[:, 1]

                        intersections_2d = np.vstack([X_proj_inter, Y_proj_inter]).T
                        intersections_rotated = (rotation_combined @ (intersections_2d - np.array([center_x_proj, center_y_proj]).T).T).T + np.array([center_x_proj, center_y_proj])
                        X_inter_rotated, Y_inter_rotated = intersections_rotated[:, 0], intersections_rotated[:, 1]

                        # 4.5 Auxiliary Lines and Proportions
                        # ----------------------------------------------------------------------------------
                        x_range_proj = np.linspace(min(X_proj_ellipse), max(X_proj_ellipse), 200)
                        if slope_traj != float('inf'):
                            y_range_proj = center_y_proj + slope_traj * (x_range_proj - center_x_proj)
                            x_shifted = x_range_proj - xc_proj
                            y_shifted = y_range_proj - yc_proj
                            x_rot = x_shifted * np.cos(-theta_proj) - y_shifted * np.sin(-theta_proj)
                            y_rot = x_shifted * np.sin(-theta_proj) + y_shifted * np.cos(-theta_proj)
                            inside_ellipse = (x_rot / a_proj)**2 + (y_rot / b_proj)**2 <= 1
                            x_range_proj = x_range_proj[inside_ellipse]
                            y_range_proj = y_range_proj[inside_ellipse]
                        else:
                            y_range_proj = np.full_like(x_range_proj, center_y_proj)
                            y_shifted = y_range_proj - yc_proj
                            x_shifted = x_range_proj - xc_proj
                            x_rot = x_shifted * np.cos(-theta_proj) - y_shifted * np.sin(-theta_proj)
                            y_rot = x_shifted * np.sin(-theta_proj) + y_shifted * np.cos(-theta_proj)
                            inside_ellipse = (x_rot / a_proj)**2 + (y_rot / b_proj)**2 <= 1
                            x_range_proj = x_range_proj[inside_ellipse]
                            y_range_proj = y_range_proj[inside_ellipse]

                        slope_traj_rotated = 0
                        x_range_rotated = np.linspace(min(X_ellipse_rotated), max(X_ellipse_rotated), 200)
                        y_range_rotated = np.full_like(x_range_rotated, center_y_proj)

                        ellipse_model_rotated = EllipseModel()
                        ellipse_points_rotated = np.column_stack((X_ellipse_rotated, Y_ellipse_rotated))
                        ellipse_model_rotated.estimate(ellipse_points_rotated)
                        xc_rotated, yc_rotated, a_rotated, b_rotated, theta_rotated = ellipse_model_rotated.params
                        x_shifted_rot = x_range_rotated - xc_rotated
                        y_shifted_rot = y_range_rotated - yc_rotated
                        x_rot_rot = x_shifted_rot * np.cos(-theta_rotated) - y_shifted_rot * np.sin(-theta_rotated)
                        y_rot_rot = x_shifted_rot * np.sin(-theta_rotated) + y_shifted_rot * np.cos(-theta_rotated)
                        inside_ellipse_rot = (x_rot_rot / a_rotated)**2 + (y_rot_rot / b_rotated)**2 <= 1
                        x_range_rotated = x_range_rotated[inside_ellipse_rot]
                        y_range_rotated = y_range_rotated[inside_ellipse_rot]

                        highest_idx = np.argmax(Y_ellipse_rotated)
                        highest_x = X_ellipse_rotated[highest_idx]
                        highest_y = Y_ellipse_rotated[highest_idx]
                        y_range_vertical = np.linspace(min(Y_ellipse_rotated), max(Y_ellipse_rotated), 200)
                        x_vertical = np.full_like(y_range_vertical, highest_x)
                        x_shifted_vertical = x_vertical - xc_rotated
                        y_shifted_vertical = y_range_vertical - yc_rotated
                        x_rot_vertical = x_shifted_vertical * np.cos(-theta_rotated) - y_shifted_vertical * np.sin(-theta_rotated)
                        y_rot_vertical = x_shifted_vertical * np.sin(-theta_rotated) + y_shifted_vertical * np.cos(-theta_rotated)
                        inside_ellipse_vertical = (x_rot_vertical / a_rotated)**2 + (y_rot_vertical / b_rotated)**2 <= 1
                        x_vertical = x_vertical[inside_ellipse_vertical]
                        y_range_vertical = y_range_vertical[inside_ellipse_vertical]

                        y_top = highest_y
                        y_trajectory = Y_traj_rotated[0]
                        y_parallel = center_y_proj
                        distance_top_to_trajectory = y_top - y_trajectory
                        distance_trajectory_to_parallel = y_trajectory - y_parallel
                        total_distance = y_top - y_parallel
                        prop_top_to_trajectory = (distance_top_to_trajectory / total_distance) * 100 if total_distance != 0 else 0
                        prop_trajectory_to_parallel = (distance_trajectory_to_parallel / total_distance) * 100 if total_distance != 0 else 0

                        # # 4.6 Visualization: Projected and Rotated Ellipse
                        # # ----------------------------------------------------------------------------------
                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                        # # 4.6.1 Original Projected Ellipse
                        # # ----------------------------------------------------------------------------------
                        # ax1.plot(X_proj_ellipse, Y_proj_ellipse, 'b-', linewidth=2, label="Projected Ellipse")
                        # ax1.scatter(X_proj_ellipse, Y_proj_ellipse, c='orange', s=20, label="Ellipse Points")
                        # ax1.scatter(X_proj_inter, Y_proj_inter, c='red', s=50, marker='o', label="Intersections")
                        # ax1.scatter(X_proj_traj, Y_proj_traj, c='green', s=10, label="Projected Trajectory")
                        # ax1.plot(x_range_proj, y_range_proj, '--', color='black', linewidth=1.0, label="Parallel Axis")
                        # ax1.set_xlabel("X_proj_ellipse")
                        # ax1.set_ylabel("Y_proj_ellipse")
                        # ax1.set_title("Original Projected Ellipse: X vs Y")
                        # ax1.grid(True)
                        # ax1.legend()
                        # ax1.axis('equal')

                        # # 4.6.2 Rotated Ellipse with Horizontal Trajectory
                        # # ----------------------------------------------------------------------------------
                        # ax2.plot(X_ellipse_rotated, Y_ellipse_rotated, 'b-', linewidth=2, label="Rotated Ellipse")
                        # ax2.scatter(X_inter_rotated, Y_inter_rotated, c='red', s=50, marker='o', label="Intersections")
                        # ax2.plot(X_traj_rotated, Y_traj_rotated, 'g-', linewidth=4, label="Horizontal Trajectory")
                        # ax2.plot(x_range_rotated, y_range_rotated, '--', color='black', linewidth=1.0, label="Parallel Axis")
                        # ax2.plot(x_vertical, y_range_vertical, '--', color='black', linewidth=1.0, label="Vertical Axis at Highest Point")
                        # x_annotate = max(X_ellipse_rotated) + 0.1 * (max(X_ellipse_rotated) - min(X_ellipse_rotated))
                        # y_mid_top_to_traj = (y_top + y_trajectory) / 2
                        # y_mid_traj_to_parallel = (y_trajectory + y_parallel) / 2
                        # ax2.text(x_annotate, y_mid_top_to_traj, f'{prop_top_to_trajectory:.2f}%', fontsize=10, color='black', verticalalignment='center')
                        # ax2.text(x_annotate, y_mid_traj_to_parallel, f'{prop_trajectory_to_parallel:.2f}%', fontsize=10, color='black', verticalalignment='center')
                        # ax2.set_xlabel("X_rotated")
                        # ax2.set_ylabel("Y_rotated")
                        # ax2.set_title("Rotated Ellipse with Horizontal Trajectory")
                        # ax2.grid(True)
                        # ax2.legend()
                        # ax2.axis('equal')

                        # plt.tight_layout()
                        # st.pyplot(fig)

                        # # 4.7 Proportions Output
                        # # ----------------------------------------------------------------------------------
                        # st.write("--- Proportions in Rotated Figure ---")
                        # st.write(f"Distance from top to trajectory: {distance_top_to_trajectory:.2e}")
                        # st.write(f"Distance from trajectory to parallel axis: {distance_trajectory_to_parallel:.2e}")
                        # st.write(f"Total distance: {total_distance:.2e}")
                        # st.write(f"Proportion (top to trajectory): {prop_top_to_trajectory:.2f}%")
                        # st.write(f"Proportion (trajectory to parallel): {prop_trajectory_to_parallel:.2f}%")

                        # # ----------------------------------------------------------------------------------
                        # # 5. Express in Local Cartesian Coordinates, well orientated
                        # # ----------------------------------------------------------------------------------
                        # # Streamlit app title
                        # st.title("Projection and Rotation in Local Coordinates")

                        # 5A. Projected Trajectory in Local Coordinates
                        # ----------------------------------------------------------------------------------
                        local_coords = np.dot(rotation_matrix_inv, projected_trajectory.T).T
                        x_local = local_coords[:, 0]
                        y_local = local_coords[:, 1]
                        z_local = local_coords[:, 2]

                        r_vals_L = np.sqrt(x_local**2 + y_local**2)
                        phi_vals_L = np.arctan2(y_local, x_local)

                        # Intersection points in local coordinates
                        intersections_local = np.dot(rotation_matrix_inv, intersection_points.T).T
                        x_inter_local = intersections_local[:, 0]
                        y_inter_local = intersections_local[:, 1]

                        # 5B. Projected Cotour in Local Coordinates
                        # ----------------------------------------------------------------------------------
                        ellipse_3d = np.vstack([X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled]).T
                        ellipse_local = np.dot(rotation_matrix_inv, ellipse_3d.T).T
                        x_ellipse_local = ellipse_local[:, 0]
                        y_ellipse_local = ellipse_local[:, 1]

                        # 5C. Rotation over everything to align the trajectory horizontally
                        # ----------------------------------------------------------------------------------
                        # Calculate the angle of the projected trajectory in local coordinates
                        dx = x_local[-1] - x_local[0]
                        dy = y_local[-1] - y_local[0]
                        trajectory_angle = np.arctan2(dy, dx) + np.pi

                        # 2D rotation matrix to align the cut with the X-axis
                        rotation_2d = np.array([
                            [np.cos(-trajectory_angle), -np.sin(-trajectory_angle)],
                            [np.sin(-trajectory_angle), np.cos(-trajectory_angle)]
                        ])

                        # Rotate the ellipse in local coordinates
                        ellipse_local_2d = np.vstack([x_ellipse_local, y_ellipse_local]).T
                        ellipse_rotated = (rotation_2d @ ellipse_local_2d.T).T
                        x_ellipse_rotated = ellipse_rotated[:, 0]
                        y_ellipse_rotated = ellipse_rotated[:, 1]

                        # Rotate the trajectory in local coordinates
                        trajectory_local_2d = np.vstack([x_local, y_local]).T
                        trajectory_rotated = (rotation_2d @ trajectory_local_2d.T).T
                        x_traj_rotated = trajectory_rotated[:, 0]
                        y_traj_rotated = trajectory_rotated[:, 1]

                        # Rotate the intersection points
                        intersections_rotated = (rotation_2d @ np.vstack([x_inter_local, y_inter_local])).T
                        x_inter_rotated = intersections_rotated[:, 0]
                        y_inter_rotated = intersections_rotated[:, 1]

                        # Important for later!! (we change the names to remember better)
                        # ----------------------------------------------------------------------------------            
                        original_x_traj = x_traj_rotated 
                        x_traj = original_x_traj  # Updated to use the desired trajectory
                        y_traj = y_traj_rotated

                        # # ----------------------------------------------------------------------------------
                        # # 6. 3D Representation
                        # # ----------------------------------------------------------------------------------
                        
                        # ----------------------------------------------------------------------------------
                        # 7. Extraction of r_vals and phi_vals from Transversal Section
                        # ----------------------------------------------------------------------------------
                        r_vals = np.sqrt(X_traj_rotated**2 + Y_traj_rotated**2)
                        phi_vals = np.arctan2(Y_traj_rotated, X_traj_rotated)
                        phi_vals = np.mod(phi_vals + 2 * np.pi, 2 * np.pi)

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
                            return r, alpha

                        r_elliptic, phi_elliptic = elliptical_coords(X_traj_rotated, Y_traj_rotated,
                                                                    xc_rotated, yc_rotated,
                                                                    a_rotated, b_rotated, theta_rotated)
                        # --- Parameters from the rotated ellipse ---
                        xc = np.mean(x_ellipse_rotated)
                        yc = np.mean(y_ellipse_rotated)
                        # Estimate semi-axes and rotation angle
                        ellipse_local_2d_rotated = np.column_stack((x_ellipse_rotated - xc, y_ellipse_rotated - yc))
                        ellipse_model = EllipseModel()
                        ellipse_model.estimate(ellipse_local_2d_rotated)
                        _, _, a_ellipse, b_ellipse, theta_ellipse = ellipse_model.params
                        delta = b_ellipse / a_ellipse
                        r_traj, alpha_traj = elliptical_coords(x_traj_rotated, y_traj_rotated, xc, yc, a_ellipse, b_ellipse, theta_ellipse)

                        # alpha_traj_original = alpha_traj


                        # ----------------------------------------------------------------------------------
                        # 8. Express the in-situ data in the Local Cartesian Coordinate System
                        # ----------------------------------------------------------------------------------
                        # Transform GSE data to Local Cartesian coordinates using the inverse rotation matrix
                        Bx_exp = Bx_exp[::-1]
                        By_exp = By_exp[::-1]
                        Bz_exp = Bz_exp[::-1]
                        
                        B_gse_exp = np.vstack([Bx_exp, By_exp, Bz_exp])
                        B_local_exp = np.dot(rotation_matrix_inv, B_gse_exp)

                        # Extract Local Cartesian components
                        Bx_local_exp = B_local_exp[0]  # B_x^L (local x-component)
                        By_local_exp = B_local_exp[1]  # B_y^L (local y-component)
                        Bz_local_exp = B_local_exp[2]  # B_z^L (local z-component)

                        # Compute the total magnetic field magnitude in Local Cartesian coordinates
                        B_total_local_exp = np.sqrt(Bx_local_exp**2 + By_local_exp**2 + Bz_local_exp**2)


                        # ----------------------------------------------------------------------------------
                        # 9. Express the in-situ data in the Local CYLINDRICAL Coordinate System
                        # ----------------------------------------------------------------------------------
                        a_ell = 1

                        # Compute metric components along the trajectory
                        grr_traj = a_ell**2 * (np.cos(alpha_traj)**2 + delta**2 * np.sin(alpha_traj)**2)
                        gyy_traj = np.ones_like(r_traj)
                        gphiphi_traj = a_ell**2 * r_traj**2 * (np.sin(alpha_traj)**2 + delta**2 * np.cos(alpha_traj)**2)
                        grphi_traj = a_ell**2 * r_traj * np.sin(alpha_traj) * np.cos(alpha_traj) * (delta**2 - 1)

                        # --- Use the transformed local components (exported data) ---
                        # Assuming B_local_exp is provided from section 8
                        Bx_local_exp = B_local_exp[0]  # B_x^L (exported)
                        By_local_exp = B_local_exp[1]  # B_y^L (exported)
                        Bz_local_exp = B_local_exp[2]  # B_z^L (exported)

                        # Reverse alpha_traj to match original convention (do this once)
                        # alpha_traj = alpha_traj[::-1]  # Ensure consistency (commented out as per your code)

                        # --- Define the transformation coefficients ---
                        cos_alpha = np.cos(alpha_traj)
                        sin_alpha = np.sin(alpha_traj)
                        coeff1 = a_ell * cos_alpha          # a * cos(φ)
                        coeff2 = -a_ell * r_traj * sin_alpha  # -a * r * sin(φ)
                        coeff3 = a_ell * delta * sin_alpha    # a * δ * sin(φ)
                        coeff4 = a_ell * delta * r_traj * cos_alpha  # a * δ * r * cos(φ)

                        # --- Transform to cylindrical coordinates by solving the system ---
                        Br_exp = np.zeros_like(r_traj)
                        Bphi_exp = np.zeros_like(r_traj)
                        determinants = []  # To track determinant stability

                        for i in range(len(r_traj)):
                            A_matrix = np.array([[coeff1[i], coeff2[i]], [coeff3[i], coeff4[i]]])
                            det = np.linalg.det(A_matrix)
                            determinants.append(det)  # Store determinant for diagnostics
                            b_vector = np.array([Bx_local_exp[i], Bz_local_exp[i]])
                            if np.abs(det) > 1e-10:  # Check for invertibility
                                Br_Bphi = np.linalg.solve(A_matrix, b_vector)
                                Br_exp[i] = Br_Bphi[0]  # B^r
                                Bphi_exp[i] = Br_Bphi[1]  # B^φ
                            else:
                                Br_exp[i] = 0
                                Bphi_exp[i] = 0
                                # st.warning(f"Singular matrix at index {i}, det = {det:.6e}")
                        By_exp = By_local_exp  # B^y = B_y in local coordinates

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

                        # 10.A. General Form Model
                        #------------------------------------------------------------------------------------------------
                        # Adjusted model functions to match the provided equations
                        def model_Br_general(r_alpha, A1_r, B1_r, C1_r, D1_r, E1_r, alpha0_r):
                            """
                            Radial component with linear and quadratic radial terms and oscillatory part
                            """
                            r, alpha = r_alpha
                            linear_part = A1_r * r
                            quadratic_part = B1_r * r**2
                            oscillatory_part = C1_r + D1_r * np.sin(alpha - alpha0_r) + E1_r * np.cos(alpha - alpha0_r)
                            return linear_part + quadratic_part * oscillatory_part

                        def model_By_general(r_alpha, A_y, B_y, C_y, D_y, alpha0_y, E_y):
                            """
                            Y component with quadratic radial term, oscillatory part, and offset
                            """
                            r, alpha = r_alpha
                            radial_part = A_y * r**2
                            oscillatory_part = B_y + C_y * np.sin(alpha - alpha0_y) + D_y * np.cos(alpha - alpha0_y)
                            return radial_part * oscillatory_part + E_y

                        def model_Bphi_general(r_alpha, A1_phi, B1_phi, C1_phi, D1_phi, E1_phi, alpha0_phi):
                            """
                            Azimuthal component with linear and quadratic radial terms and oscillatory part
                            """
                            r, alpha = r_alpha
                            linear_part = A1_phi * r
                            quadratic_part = B1_phi * r**2
                            oscillatory_part = C1_phi + D1_phi * np.sin(alpha - alpha0_phi) + E1_phi * np.cos(alpha - alpha0_phi)
                            return linear_part + quadratic_part * oscillatory_part

                        # Preparación de datos (manteniendo tu estructura original)
                        r_vals_local_eliptic_general, alpha_traj_elliptic_general = elliptical_coords(x_traj_rotated, y_traj_rotated, xc, yc, a_ellipse, b_ellipse, theta_ellipse)
                        alpha_traj_elliptic_general = alpha_traj
                        data_fit_general = np.vstack((r_vals_local_eliptic_general, alpha_traj_elliptic_general)).T

                        # Valores iniciales para el ajuste (6 parámetros por componente)
                        initial_guess_Br_general = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # [A1_r, B1_r, C1_r, D1_r, E1_r, alpha0_r]
                        initial_guess_By_general = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]   # [A_y, B_y, C_y, D_y, alpha0_y, E_y]
                        initial_guess_Bphi_general = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0] # [A1_phi, B1_phi, C1_phi, D1_phi, E1_phi, alpha0_phi]

                        try:
                            # Fit for Br
                            params_Br_general, _ = curve_fit(model_Br_general, data_fit_general.T, Br_exp, p0=initial_guess_Br_general)
                            A1_Br_general, B1_Br_general, C1_Br_general, D1_Br_general, E1_Br_general, alpha0_Br_general = params_Br_general
                            Br_vector_general = model_Br_general(data_fit_general.T, A1_Br_general, B1_Br_general, C1_Br_general, D1_Br_general, E1_Br_general, alpha0_Br_general)

                            # Fit for By
                            params_By_general, _ = curve_fit(model_By_general, data_fit_general.T, By_exp, p0=initial_guess_By_general)
                            A_By_general, B_By_general, C_By_general, D_By_general, alpha0_By_general, E_By_general = params_By_general
                            By_vector_general = model_By_general(data_fit_general.T, A_By_general, B_By_general, C_By_general, D_By_general, alpha0_By_general, E_By_general)

                            # Fit for Bphi
                            params_Bphi_general, _ = curve_fit(model_Bphi_general, data_fit_general.T, Bphi_exp, p0=initial_guess_Bphi_general)
                            A1_Bphi_general, B1_Bphi_general, C1_Bphi_general, D1_Bphi_general, E1_Bphi_general, alpha0_Bphi_general = params_Bphi_general
                            Bphi_vector_general = model_Bphi_general(data_fit_general.T, A1_Bphi_general, B1_Bphi_general, C1_Bphi_general, D1_Bphi_general, E1_Bphi_general, alpha0_Bphi_general)

                            # R² calculations
                            ss_tot_Br_general = np.sum((Br_exp - np.mean(Br_exp))**2)
                            ss_res_Br_general = np.sum((Br_exp - Br_vector_general)**2)
                            R2_Br_general = 1 - (ss_res_Br_general / ss_tot_Br_general) if ss_tot_Br_general != 0 else 0

                            ss_tot_By_general = np.sum((By_exp - np.mean(By_exp))**2)
                            ss_res_By_general = np.sum((By_exp - By_vector_general)**2)
                            R2_By_general = 1 - (ss_res_By_general / ss_tot_By_general) if ss_tot_By_general != 0 else 0

                            ss_tot_Bphi_general = np.sum((Bphi_exp - np.mean(Bphi_exp))**2)
                            ss_res_Bphi_general = np.sum((Bphi_exp - Bphi_vector_general)**2)
                            R2_Bphi_general = 1 - (ss_res_Bphi_general / ss_tot_Bphi_general) if ss_tot_Bphi_general != 0 else 0

                            R2_avg_general = (R2_Br_general + R2_By_general + R2_Bphi_general) / 3
                            
                            # Magnitud total del campo magnético
                            B_vector_general = np.sqrt(
                                grr_traj * Br_vector_general**2 +
                                gyy_traj * By_vector_general**2 +
                                gphiphi_traj * Bphi_vector_general**2 +
                                2 * grphi_traj * Br_vector_general * Bphi_vector_general
                            )

                        except RuntimeError:
                            st.write("Error en el ajuste de curvas: no se pudo encontrar una solución óptima")

                        # ----------------------------------------------------------------------------------
                        # 11.A Go back to local and then GSE coordinate system (for the fitted values)
                        # ----------------------------------------------------------------------------------

                        # --- Transformación a coordenadas cartesianas locales ---
                        x_traj_general = original_x_traj  # Usamos la trayectoria original en X
                        y_traj_general = y_traj_rotated

                        Bx_traj_general = Br_vector_general * a_ell * np.cos(alpha_traj) - Bphi_vector_general * a_ell * r_vals_local_eliptic_general * np.sin(alpha_traj)
                        By_traj_cartesian_general = By_vector_general  # By ya está alineado con el eje y local
                        Bz_traj_general = Br_vector_general * a_ell * delta * np.sin(alpha_traj) + Bphi_vector_general * delta * a_ell * r_vals_local_eliptic_general * np.cos(alpha_traj)
                        B_vector_cartesian_general = np.sqrt(Bx_traj_general**2 + By_traj_cartesian_general**2 + Bz_traj_general**2)

                        # --- Transformación a coordenadas GSE (requiere rotation_matrix) ---
                        try:
                            B_local_general = np.vstack((Bx_traj_general, By_traj_cartesian_general, Bz_traj_general))  # Shape: (3, N_points)
                            B_GSE_general = rotation_matrix @ B_local_general  # Shape: (3, N_points)
                            Bx_GSE_general = B_GSE_general[0, :]
                            By_GSE_general = B_GSE_general[1, :]
                            Bz_GSE_general = B_GSE_general[2, :]
                        except NameError:
                            st.warning("rotation_matrix no está definida. Mostrando solo coordenadas cartesianas locales.")
                            Bx_GSE_general, By_GSE_general, Bz_GSE_general = None, None, None

                        # ----------------------------------------------------------------------------------
                        # 12.A Compute quality factor for the original in-situ data
                        # ----------------------------------------------------------------------------------

                        # Slice original data to match the fitted segment
                        B_data_segment_general = B_data[initial_point - 1:final_point - 1][::-1]
                        Bx_data_segment_general = Bx_data[initial_point - 1:final_point - 1][::-1]
                        By_data_segment_general = By_data[initial_point - 1:final_point - 1][::-1]
                        Bz_data_segment_general = Bz_data[initial_point - 1:final_point - 1][::-1]

                        # Cálculo del coeficiente de determinación R² (goodness-of-fit)
                        # Para la intensidad total del campo magnético (B)
                        ss_tot_B_general = np.sum((B_data_segment_general - np.mean(B_data_segment_general))**2)
                        ss_res_B_general = np.sum((B_data_segment_general - B_vector_general)**2)
                        R2_B_general = 1 - (ss_res_B_general / ss_tot_B_general) if ss_tot_B_general != 0 else 0

                        # Para la componente Bx
                        ss_tot_Bx_general = np.sum((Bx_data_segment_general - np.mean(Bx_data_segment_general))**2)
                        ss_res_Bx_general = np.sum((Bx_data_segment_general - Bx_GSE_general)**2)
                        R2_Bx_general = 1 - (ss_res_Bx_general / ss_tot_Bx_general) if ss_tot_Bx_general != 0 else 0

                        # Para la componente By
                        ss_tot_By_general = np.sum((By_data_segment_general - np.mean(By_data_segment_general))**2)
                        ss_res_By_general = np.sum((By_data_segment_general - By_GSE_general)**2)
                        R2_By_general = 1 - (ss_res_By_general / ss_tot_By_general) if ss_tot_By_general != 0 else 0

                        # Para la componente Bz
                        ss_tot_Bz_general = np.sum((Bz_data_segment_general - np.mean(Bz_data_segment_general))**2)
                        ss_res_Bz_general = np.sum((Bz_data_segment_general - Bz_GSE_general)**2)
                        R2_Bz_general = 1 - (ss_res_Bz_general / ss_tot_Bz_general) if ss_tot_Bz_general != 0 else 0

                        # Cálculo del R² promedio
                        R2_avg_general = (R2_B_general + R2_Bx_general + R2_By_general + R2_Bz_general) / 4



                        # 10.B. Only Radial Dependency Model
                        #------------------------------------------------------------------------------------------------

                        def model_Br_radial(r_alpha, a):
                            """
                            Componente radial fija en cero
                            """
                            a = 0
                            r, alpha = r_alpha
                            return np.zeros_like(r)

                        def model_By_radial(r_alpha, A, B):
                            """
                            Componente Y con dependencia cuadrática radial
                            """
                            r, alpha = r_alpha
                            return A + B * r**2

                        def model_Bphi_radial(r_alpha, C):
                            """
                            Componente azimutal con dependencia lineal
                            """
                            r, alpha = r_alpha
                            return C * r

                        # Preparación de datos (manteniendo tu estructura original)
                        r_vals_local_eliptic_radial, alpha_traj_elliptic_radial = elliptical_coords(x_traj_rotated, y_traj_rotated, xc, yc, a_ellipse, b_ellipse, theta_ellipse)
                        alpha_traj_elliptic_radial = alpha_traj
                        data_fit_radial = np.vstack((r_vals_local_eliptic_radial, alpha_traj_elliptic_radial)).T

                        # Valores iniciales para el ajuste
                        initial_guess_Br_radial = [0.0]  # Para Br (aunque está fijo en 0)
                        initial_guess_By_radial = [1.0, 1.0]  # [A, B]
                        initial_guess_Bphi_radial = [1.0]  # [C]

                        try:
                            # Ajuste de curvas para cada componente
                            params_Br_radial, _ = curve_fit(model_Br_radial, data_fit_radial.T, Br_exp, p0=initial_guess_Br_radial)
                            a_Br_radial = params_Br_radial[0]  # Aunque el modelo fuerza a = 0, curve_fit devuelve un valor
                            Br_vector_radial = model_Br_radial(data_fit_radial.T, a_Br_radial)

                            params_By_radial, _ = curve_fit(model_By_radial, data_fit_radial.T, By_exp, p0=initial_guess_By_radial)
                            A_By_radial, B_By_radial = params_By_radial
                            By_vector_radial = model_By_radial(data_fit_radial.T, A_By_radial, B_By_radial)

                            params_Bphi_radial, _ = curve_fit(model_Bphi_radial, data_fit_radial.T, Bphi_exp, p0=initial_guess_Bphi_radial)
                            C_Bphi_radial = params_Bphi_radial[0]
                            Bphi_vector_radial = model_Bphi_radial(data_fit_radial.T, C_Bphi_radial)

                            # Cálculo de R² para cada componente
                            # Nota: R² para Br no tiene sentido porque Br_exp se ajusta a 0, pero lo calculamos para consistencia
                            ss_tot_Br_radial = np.sum((Br_exp - np.mean(Br_exp))**2)
                            ss_res_Br_radial = np.sum((Br_exp - Br_vector_radial)**2)
                            R2_Br_radial = 1 - (ss_res_Br_radial / ss_tot_Br_radial) if ss_tot_Br_radial != 0 else 0

                            ss_tot_By_radial = np.sum((By_exp - np.mean(By_exp))**2)
                            ss_res_By_radial = np.sum((By_exp - By_vector_radial)**2)
                            R2_By_radial = 1 - (ss_res_By_radial / ss_tot_By_radial) if ss_tot_By_radial != 0 else 0

                            ss_tot_Bphi_radial = np.sum((Bphi_exp - np.mean(Bphi_exp))**2)
                            ss_res_Bphi_radial = np.sum((Bphi_exp - Bphi_vector_radial)**2)
                            R2_Bphi_radial = 1 - (ss_res_Bphi_radial / ss_tot_Bphi_radial) if ss_tot_Bphi_radial != 0 else 0

                            R2_avg_radial = (R2_Br_radial + R2_By_radial + R2_Bphi_radial) / 3  # Incluye Br para consistencia, aunque sea trivial

                            # Vectores ajustados (ya calculados, pero renombrados para claridad)
                            Br_vector_radial = model_Br_radial(data_fit_radial.T, a_Br_radial)
                            By_vector_radial = model_By_radial(data_fit_radial.T, A_By_radial, B_By_radial)
                            Bphi_vector_radial = model_Bphi_radial(data_fit_radial.T, C_Bphi_radial)

                            B_vector_radial = np.sqrt(
                                grr_traj * Br_vector_radial**2 +
                                gyy_traj * By_vector_radial**2 +
                                gphiphi_traj * Bphi_vector_radial**2 +
                                2 * grphi_traj * Br_vector_radial * Bphi_vector_radial
                            )

                        except RuntimeError:
                            st.write("Error en el ajuste de curvas: no se pudo encontrar una solución óptima")

                        # ----------------------------------------------------------------------------------
                        # 11.B Go back to local and then GSE coordinate system (for the fitted values)
                        # ----------------------------------------------------------------------------------

                        # --- Transformación a coordenadas cartesianas locales ---
                        x_traj_radial = original_x_traj  # Usamos la trayectoria original en X
                        y_traj_radial = y_traj_rotated

                        Bx_traj_radial = Br_vector_radial * a_ell * np.cos(alpha_traj) - Bphi_vector_radial * a_ell * r_vals_local_eliptic_radial * np.sin(alpha_traj)
                        By_traj_cartesian_radial = By_vector_radial  # By ya está alineado con el eje y local
                        Bz_traj_radial = Br_vector_radial * a_ell * delta * np.sin(alpha_traj) + Bphi_vector_radial * delta * a_ell * r_vals_local_eliptic_radial * np.cos(alpha_traj)
                        B_vector_cartesian_radial = np.sqrt(Bx_traj_radial**2 + By_traj_cartesian_radial**2 + Bz_traj_radial**2)

                        # --- Transformación a coordenadas GSE (requiere rotation_matrix) ---
                        try:
                            B_local_radial = np.vstack((Bx_traj_radial, By_traj_cartesian_radial, Bz_traj_radial))  # Shape: (3, N_points)
                            B_GSE_radial = rotation_matrix @ B_local_radial  # Shape: (3, N_points)
                            Bx_GSE_radial = B_GSE_radial[0, :]
                            By_GSE_radial = B_GSE_radial[1, :]
                            Bz_GSE_radial = B_GSE_radial[2, :]
                        except NameError:
                            st.warning("rotation_matrix no está definida. Mostrando solo coordenadas cartesianas locales.")
                            Bx_GSE_radial, By_GSE_radial, Bz_GSE_radial = None, None, None

                        # ----------------------------------------------------------------------------------
                        # 12.B Compute quality factor for the original in-situ data
                        # ----------------------------------------------------------------------------------

                        # Slice original data to match the fitted segment
                        B_data_segment_radial = B_data[initial_point - 1:final_point - 1][::-1]
                        Bx_data_segment_radial = Bx_data[initial_point - 1:final_point - 1][::-1]
                        By_data_segment_radial = By_data[initial_point - 1:final_point - 1][::-1]
                        Bz_data_segment_radial = Bz_data[initial_point - 1:final_point - 1][::-1]

                        # Cálculo del coeficiente de determinación R² (goodness-of-fit)
                        # Para la intensidad total del campo magnético (B)
                        ss_tot_B_radial = np.sum((B_data_segment_radial - np.mean(B_data_segment_radial))**2)
                        ss_res_B_radial = np.sum((B_data_segment_radial - B_vector_radial)**2)
                        R2_B_radial = 1 - (ss_res_B_radial / ss_tot_B_radial) if ss_tot_B_radial != 0 else 0

                        # Para la componente Bx
                        ss_tot_Bx_radial = np.sum((Bx_data_segment_radial - np.mean(Bx_data_segment_radial))**2)
                        ss_res_Bx_radial = np.sum((Bx_data_segment_radial - Bx_GSE_radial)**2)
                        R2_Bx_radial = 1 - (ss_res_Bx_radial / ss_tot_Bx_radial) if ss_tot_Bx_radial != 0 else 0

                        # Para la componente By
                        ss_tot_By_radial = np.sum((By_data_segment_radial - np.mean(By_data_segment_radial))**2)
                        ss_res_By_radial = np.sum((By_data_segment_radial - By_GSE_radial)**2)
                        R2_By_radial = 1 - (ss_res_By_radial / ss_tot_By_radial) if ss_tot_By_radial != 0 else 0

                        # Para la componente Bz
                        ss_tot_Bz_radial = np.sum((Bz_data_segment_radial - np.mean(Bz_data_segment_radial))**2)
                        ss_res_Bz_radial = np.sum((Bz_data_segment_radial - Bz_GSE_radial)**2)
                        R2_Bz_radial = 1 - (ss_res_Bz_radial / ss_tot_Bz_radial) if ss_tot_Bz_radial != 0 else 0

                        # Cálculo del R² promedio
                        R2_avg_radial = (R2_B_radial + R2_Bx_radial + R2_By_radial + R2_Bz_radial) / 4


                        # 10.C. Oscillatory Model with Radial Dependency
                        # ----------------------------------------------------------------------------------

                        # 10.C.1. Models for the fitting
                        #------------------------------------------------------------------------------------------------
                        def model_Br_oscillatory(r_alpha, a):
                            """
                            Componente radial fija en cero
                            """
                            a = 0
                            r, alpha = r_alpha
                            return np.zeros_like(r)

                        def model_By_oscillatory(r_alpha, A, B, C, D, alpha_0, E):
                            """
                            Componente Y con dependencia cuadrática radial y término oscilatorio
                            """
                            r, alpha = r_alpha
                            radial_part = A * r**2
                            oscillatory_part = B + C * np.sin(alpha - alpha_0) + D * np.cos(alpha - alpha_0)
                            return radial_part * oscillatory_part + E

                        def model_Bphi_oscillatory(r_alpha, A1, B1, C1):
                            """
                            Componente azimutal con dependencia polinómica cúbica
                            """
                            r, alpha = r_alpha
                            return A1 * r + B1 * r**2 + C1 * r**3

                        # Preparación de datos (manteniendo tu estructura original)
                        r_vals_local_eliptic_oscillatory, alpha_traj_elliptic_oscillatory = elliptical_coords(x_traj_rotated, y_traj_rotated, xc, yc, a_ellipse, b_ellipse, theta_ellipse)
                        alpha_traj_elliptic_oscillatory = alpha_traj
                        data_fit_oscillatory = np.vstack((r_vals_local_eliptic_oscillatory, alpha_traj_elliptic_oscillatory)).T

                        # Valores iniciales para el ajuste
                        initial_guess_Br_oscillatory = [0.0]  # Para Br (aunque está fijo en 0)
                        initial_guess_By_oscillatory = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # [A, B, C, D, alpha_0, E]
                        initial_guess_Bphi_oscillatory = [1.0, 1.0, 1.0]  # [A1, B1, C1]

                        try:
                            # Ajuste de curvas para cada componente
                            params_Br_oscillatory, _ = curve_fit(model_Br_oscillatory, data_fit_oscillatory.T, Br_exp, p0=initial_guess_Br_oscillatory)
                            a_Br_oscillatory = params_Br_oscillatory[0]  # Aunque el modelo fuerza a = 0
                            Br_vector_oscillatory = model_Br_oscillatory(data_fit_oscillatory.T, a_Br_oscillatory)

                            params_By_oscillatory, _ = curve_fit(model_By_oscillatory, data_fit_oscillatory.T, By_exp, p0=initial_guess_By_oscillatory)
                            A_By_oscillatory, B_By_oscillatory, C_By_oscillatory, D_By_oscillatory, alpha0_By_oscillatory, E_By_oscillatory = params_By_oscillatory
                            By_vector_oscillatory = model_By_oscillatory(data_fit_oscillatory.T, A_By_oscillatory, B_By_oscillatory, C_By_oscillatory, D_By_oscillatory, alpha0_By_oscillatory, E_By_oscillatory)

                            params_Bphi_oscillatory, _ = curve_fit(model_Bphi_oscillatory, data_fit_oscillatory.T, Bphi_exp, p0=initial_guess_Bphi_oscillatory)
                            A1_Bphi_oscillatory, B1_Bphi_oscillatory, C1_Bphi_oscillatory = params_Bphi_oscillatory
                            Bphi_vector_oscillatory = model_Bphi_oscillatory(data_fit_oscillatory.T, A1_Bphi_oscillatory, B1_Bphi_oscillatory, C1_Bphi_oscillatory)

                            # Cálculo de R² para cada componente
                            ss_tot_Br_oscillatory = np.sum((Br_exp - np.mean(Br_exp))**2)
                            ss_res_Br_oscillatory = np.sum((Br_exp - Br_vector_oscillatory)**2)
                            R2_Br_oscillatory = 1 - (ss_res_Br_oscillatory / ss_tot_Br_oscillatory) if ss_tot_Br_oscillatory != 0 else 0

                            ss_tot_By_oscillatory = np.sum((By_exp - np.mean(By_exp))**2)
                            ss_res_By_oscillatory = np.sum((By_exp - By_vector_oscillatory)**2)
                            R2_By_oscillatory = 1 - (ss_res_By_oscillatory / ss_tot_By_oscillatory) if ss_tot_By_oscillatory != 0 else 0

                            ss_tot_Bphi_oscillatory = np.sum((Bphi_exp - np.mean(Bphi_exp))**2)
                            ss_res_Bphi_oscillatory = np.sum((Bphi_exp - Bphi_vector_oscillatory)**2)
                            R2_Bphi_oscillatory = 1 - (ss_res_Bphi_oscillatory / ss_tot_Bphi_oscillatory) if ss_tot_Bphi_oscillatory != 0 else 0

                            R2_avg_oscillatory = (R2_Br_oscillatory + R2_By_oscillatory + R2_Bphi_oscillatory) / 3  # Incluye Br para consistencia

                            # Magnitud total del campo magnético
                            B_vector_oscillatory = np.sqrt(
                                grr_traj * Br_vector_oscillatory**2 +
                                gyy_traj * By_vector_oscillatory**2 +
                                gphiphi_traj * Bphi_vector_oscillatory**2 +
                                2 * grphi_traj * Br_vector_oscillatory * Bphi_vector_oscillatory
                            )

                        except RuntimeError:
                            st.write("Error en el ajuste de curvas: no se pudo encontrar una solución óptima")

                        # 11.C Go back to local and then GSE coordinate system (for the fitted values)
                        # ----------------------------------------------------------------------------------

                        # --- Transformación a coordenadas cartesianas locales ---
                        x_traj_oscillatory = original_x_traj  # Usamos la trayectoria original en X
                        y_traj_oscillatory = y_traj_rotated

                        Bx_traj_oscillatory = Br_vector_oscillatory * a_ell * np.cos(alpha_traj) - Bphi_vector_oscillatory * a_ell * r_vals_local_eliptic_oscillatory * np.sin(alpha_traj)
                        By_traj_cartesian_oscillatory = By_vector_oscillatory  # By ya está alineado con el eje y local
                        Bz_traj_oscillatory = Br_vector_oscillatory * a_ell * delta * np.sin(alpha_traj) + Bphi_vector_oscillatory * delta * a_ell * r_vals_local_eliptic_oscillatory * np.cos(alpha_traj)
                        B_vector_cartesian_oscillatory = np.sqrt(Bx_traj_oscillatory**2 + By_traj_cartesian_oscillatory**2 + Bz_traj_oscillatory**2)

                        # --- Transformación a coordenadas GSE (requiere rotation_matrix) ---
                        try:
                            B_local_oscillatory = np.vstack((Bx_traj_oscillatory, By_traj_cartesian_oscillatory, Bz_traj_oscillatory))  # Shape: (3, N_points)
                            B_GSE_oscillatory = rotation_matrix @ B_local_oscillatory  # Shape: (3, N_points)
                            Bx_GSE_oscillatory = B_GSE_oscillatory[0, :]
                            By_GSE_oscillatory = B_GSE_oscillatory[1, :]
                            Bz_GSE_oscillatory = B_GSE_oscillatory[2, :]
                        except NameError:
                            st.warning("rotation_matrix no está definida. Mostrando solo coordenadas cartesianas locales.")
                            Bx_GSE_oscillatory, By_GSE_oscillatory, Bz_GSE_oscillatory = None, None, None

                        # 12.C Compute quality factor for the original in-situ data
                        # ----------------------------------------------------------------------------------

                        # Slice original data to match the fitted segment
                        B_data_segment_oscillatory = B_data[initial_point - 1:final_point - 1][::-1]
                        Bx_data_segment_oscillatory = Bx_data[initial_point - 1:final_point - 1][::-1]
                        By_data_segment_oscillatory = By_data[initial_point - 1:final_point - 1][::-1]
                        Bz_data_segment_oscillatory = Bz_data[initial_point - 1:final_point - 1][::-1]

                        # Cálculo del coeficiente de determinación R² (goodness-of-fit)
                        # Para la intensidad total del campo magnético (B)
                        ss_tot_B_oscillatory = np.sum((B_data_segment_oscillatory - np.mean(B_data_segment_oscillatory))**2)
                        ss_res_B_oscillatory = np.sum((B_data_segment_oscillatory - B_vector_oscillatory)**2)
                        R2_B_oscillatory = 1 - (ss_res_B_oscillatory / ss_tot_B_oscillatory) if ss_tot_B_oscillatory != 0 else 0

                        # Para la componente Bx
                        ss_tot_Bx_oscillatory = np.sum((Bx_data_segment_oscillatory - np.mean(Bx_data_segment_oscillatory))**2)
                        ss_res_Bx_oscillatory = np.sum((Bx_data_segment_oscillatory - Bx_GSE_oscillatory)**2)
                        R2_Bx_oscillatory = 1 - (ss_res_Bx_oscillatory / ss_tot_Bx_oscillatory) if ss_tot_Bx_oscillatory != 0 else 0

                        # Para la componente By
                        ss_tot_By_oscillatory = np.sum((By_data_segment_oscillatory - np.mean(By_data_segment_oscillatory))**2)
                        ss_res_By_oscillatory = np.sum((By_data_segment_oscillatory - By_GSE_oscillatory)**2)
                        R2_By_oscillatory = 1 - (ss_res_By_oscillatory / ss_tot_By_oscillatory) if ss_tot_By_oscillatory != 0 else 0

                        # Para la componente Bz
                        ss_tot_Bz_oscillatory = np.sum((Bz_data_segment_oscillatory - np.mean(Bz_data_segment_oscillatory))**2)
                        ss_res_Bz_oscillatory = np.sum((Bz_data_segment_oscillatory - Bz_GSE_oscillatory)**2)
                        R2_Bz_oscillatory = 1 - (ss_res_Bz_oscillatory / ss_tot_Bz_oscillatory) if ss_tot_Bz_oscillatory != 0 else 0

                        # Cálculo del R² promedio
                        R2_avg_oscillatory = (R2_B_oscillatory + R2_Bx_oscillatory + R2_By_oscillatory + R2_Bz_oscillatory) / 4






                        # ----------------------------------------------------------------------------------
                        # Actualizar la mejor combinación si el R² actual es mejor
                        if R2_avg_general > best_R2_general:
                            best_R2_general = R2_avg_general

                            # Format parameters with 4 significant figures in scientific notation
                            A1_Br_rounded_general = f"{A1_Br_general:.4e}"
                            B1_Br_rounded_general = f"{B1_Br_general:.4e}"
                            C1_Br_rounded_general = f"{C1_Br_general:.4e}"
                            D1_Br_rounded_general = f"{D1_Br_general:.4e}"
                            alpha0_Br_rounded_general = f"{alpha0_Br_general:.4e}"
                            E1_Br_rounded_general = f"{E1_Br_general:.4e}"

                            A_By_rounded_general = f"{A_By_general:.4e}"
                            B_By_rounded_general = f"{B_By_general:.4e}"
                            C_By_rounded_general = f"{C_By_general:.4e}"
                            D_By_rounded_general = f"{D_By_general:.4e}"
                            alpha0_By_rounded_general = f"{alpha0_By_general:.4e}"
                            E_By_rounded_general = f"{E_By_general:.4e}"

                            A1_Bphi_rounded_general = f"{A1_Bphi_general:.4e}"
                            B1_Bphi_rounded_general = f"{B1_Bphi_general:.4e}"
                            C1_Bphi_rounded_general = f"{C1_Bphi_general:.4e}"
                            D1_Bphi_rounded_general = f"{D1_Bphi_general:.4e}"
                            alpha0_Bphi_rounded_general = f"{alpha0_Bphi_general:.4e}"
                            E1_Bphi_rounded_general = f"{E1_Bphi_general:.4e}"

                            params_Br_fit_general = (A1_Br_general, B1_Br_general, C1_Br_general, D1_Br_general, E1_Br_general, alpha0_Br_general)
                            params_By_fit_general = (A_By_general, B_By_general, C_By_general, D_By_general, alpha0_By_general, E_By_general)
                            params_Bphi_fit_general = (A1_Bphi_general, B1_Bphi_general, C1_Bphi_general, D1_Bphi_general, E1_Bphi_general, alpha0_Bphi_general)

                            # Define symbolic expressions using the formatted parameters
                            r, alpha = sp.symbols('r alpha')

                            # B_r expression
                            Br_expr_general = (sp.sympify(A1_Br_rounded_general) * r + sp.sympify(B1_Br_rounded_general) * r**2 * (
                                sp.sympify(C1_Br_rounded_general) +
                                sp.sympify(D1_Br_rounded_general) * sp.sin(alpha - sp.sympify(alpha0_Br_rounded_general)) +
                                sp.sympify(E1_Br_rounded_general) * sp.cos(alpha - sp.sympify(alpha0_Br_rounded_general))
                            ))

                            # B_y expression
                            By_expr_general = (sp.sympify(A_By_rounded_general) * r**2 * (
                                sp.sympify(B_By_rounded_general) +
                                sp.sympify(C_By_rounded_general) * sp.sin(alpha - sp.sympify(alpha0_By_rounded_general)) +
                                sp.sympify(D_By_rounded_general) * sp.cos(alpha - sp.sympify(alpha0_By_rounded_general))
                            )) + sp.sympify(E_By_rounded_general)

                            # B_phi expression
                            Bphi_expr_general = (sp.sympify(A1_Bphi_rounded_general) * r + sp.sympify(B1_Bphi_rounded_general) * r**2 * (
                                sp.sympify(C1_Bphi_rounded_general) +
                                sp.sympify(D1_Bphi_rounded_general) * sp.sin(alpha - sp.sympify(alpha0_Bphi_rounded_general)) +
                                sp.sympify(E1_Bphi_rounded_general) * sp.cos(alpha - sp.sympify(alpha0_Bphi_rounded_general))
                            ))

                            # SAVING VARIABLES: 
                            best_combination_general = (z0, angle_x, angle_y, angle_z, delta)

                            quality_factors_general = (R2_B_general, R2_Bx_general, R2_By_general, R2_Bz_general, R2_avg_general)
                            # Plot 1: Vertical cut in the cylinder: 
                            plot1_vars_general = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs)

                            # Plot 2: Projection and Rotation in Local Coordinates
                            plot2_vars_general = (x_local, y_local, x_ellipse_local, y_ellipse_local, x_inter_local, y_inter_local, x_ellipse_rotated, y_ellipse_rotated, x_traj_rotated, y_traj_rotated, x_inter_rotated, y_inter_rotated)

                            # Plot 3: 3D Representation
                            plot3_vars_general = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)

                            # Plot 4: Radial distance and angles in the trajectory (the 4 plots)
                            plot4_vars_general = (r_vals, phi_vals, r_elliptic, phi_elliptic, original_x_traj, alpha_traj, r_vals_local_eliptic_general, alpha_traj_elliptic_general)

                            # Plot 5: In-situ data in Local Cartesian coordinates
                            plot5_vars_general = (Bx_local_exp, By_local_exp, Bz_local_exp, B_total_local_exp)

                            # Plot 6: In-situ Cylindrical Components with fitting
                            plot6_vars_general = (Br_exp, By_exp, Bphi_exp, B_total_exp_cyl, Br_vector_general, By_vector_general, Bphi_vector_general, B_vector_general)

                            # Plot 7: Fitted local and GSE components
                            plot7_vars_general = (x_traj, B_vector_general, Bx_traj_general, By_traj_cartesian_general, Bz_traj_general, Bx_GSE_general, By_GSE_general, Bz_GSE_general)
                            
                            viz_3d_vars_opt_general = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled,
                                            scale_factor, a, Z_max_scaled, z_cut_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 
                                            X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)


                        # ----------------------------------------------------------------------------------
                        # Actualizar la mejor combinación si el R² actual es mejor
                        if R2_avg_radial > best_R2_radial:
                            best_R2_radial = R2_avg_radial

                            # Format parameters with 4 significant figures in scientific notation
                            a_Br_rounded_radial = f"{a_Br_radial:.4e}"  # For Br (fixed at 0)
                            
                            A_By_rounded_radial = f"{A_By_radial:.4e}"
                            B_By_rounded_radial = f"{B_By_radial:.4e}"

                            C_Bphi_rounded_radial = f"{C_Bphi_radial:.4e}"

                            # Parameter tuples for the radial model
                            params_Br_fit_radial = (a_Br_radial,)
                            params_By_fit_radial = (A_By_radial, B_By_radial)
                            params_Bphi_fit_radial = (C_Bphi_radial,)

                            # Define symbolic expressions using the formatted parameters
                            r, alpha = sp.symbols('r alpha')

                            # B_r expression (fixed at 0)
                            Br_expr_radial = sp.sympify(a_Br_rounded_radial)  # This will be 0

                            # B_y expression
                            By_expr_radial = (sp.sympify(A_By_rounded_radial) + sp.sympify(B_By_rounded_radial) * r**2)

                            # B_phi expression
                            Bphi_expr_radial = (sp.sympify(C_Bphi_rounded_radial) * r)

                            # SAVING VARIABLES:
                            best_combination_radial = (z0, angle_x, angle_y, angle_z, delta)

                            quality_factors_radial = (R2_B_radial, R2_Bx_radial, R2_By_radial, R2_Bz_radial, R2_avg_radial)
                            # Plot 1: Vertical cut in the cylinder:
                            plot1_vars_radial = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs)

                            # Plot 2: Projection and Rotation in Local Coordinates
                            plot2_vars_radial = (x_local, y_local, x_ellipse_local, y_ellipse_local, x_inter_local, y_inter_local, x_ellipse_rotated, y_ellipse_rotated, x_traj_rotated, y_traj_rotated, x_inter_rotated, y_inter_rotated)

                            # Plot 3: 3D Representation
                            plot3_vars_radial = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)

                            # Plot 4: Radial distance and angles in the trajectory (the 4 plots)
                            plot4_vars_radial = (r_vals, phi_vals, r_elliptic, phi_elliptic, original_x_traj, alpha_traj, r_vals_local_eliptic_radial, alpha_traj_elliptic_radial)

                            # Plot 5: In-situ data in Local Cartesian coordinates
                            plot5_vars_radial = (Bx_local_exp, By_local_exp, Bz_local_exp, B_total_local_exp)

                            # Plot 6: In-situ Cylindrical Components with fitting
                            plot6_vars_radial = (Br_exp, By_exp, Bphi_exp, B_total_exp_cyl, Br_vector_radial, By_vector_radial, Bphi_vector_radial, B_vector_radial)

                            # Plot 7: Fitted local and GSE components
                            plot7_vars_radial = (x_traj_radial, B_vector_radial, Bx_traj_radial, By_traj_cartesian_radial, Bz_traj_radial, Bx_GSE_radial, By_GSE_radial, Bz_GSE_radial)
                            
                            viz_3d_vars_opt_radial = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled,
                                                    scale_factor, a, Z_max_scaled, z_cut_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 
                                                    X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)


                        # ----------------------------------------------------------------------------------
                        # Actualizar la mejor combinación si el R² actual es mejor
                        if R2_avg_oscillatory > best_R2_oscillatory:
                            best_R2_oscillatory = R2_avg_oscillatory

                            # Format parameters with 4 significant figures in scientific notation
                            a_Br_rounded_oscillatory = f"{a_Br_oscillatory:.4e}"  # For Br (fixed at 0)
                            
                            A_By_rounded_oscillatory = f"{A_By_oscillatory:.4e}"
                            B_By_rounded_oscillatory = f"{B_By_oscillatory:.4e}"
                            C_By_rounded_oscillatory = f"{C_By_oscillatory:.4e}"
                            D_By_rounded_oscillatory = f"{D_By_oscillatory:.4e}"
                            alpha0_By_rounded_oscillatory = f"{alpha0_By_oscillatory:.4e}"
                            E_By_rounded_oscillatory = f"{E_By_oscillatory:.4e}"

                            A1_Bphi_rounded_oscillatory = f"{A1_Bphi_oscillatory:.4e}"
                            B1_Bphi_rounded_oscillatory = f"{B1_Bphi_oscillatory:.4e}"
                            C1_Bphi_rounded_oscillatory = f"{C1_Bphi_oscillatory:.4e}"

                            # Parameter tuples for the oscillatory model
                            params_Br_fit_oscillatory = (a_Br_oscillatory,)
                            params_By_fit_oscillatory = (A_By_oscillatory, B_By_oscillatory, C_By_oscillatory, D_By_oscillatory, alpha0_By_oscillatory, E_By_oscillatory)
                            params_Bphi_fit_oscillatory = (A1_Bphi_oscillatory, B1_Bphi_oscillatory, C1_Bphi_oscillatory)

                            # Define symbolic expressions using the formatted parameters
                            r, alpha = sp.symbols('r alpha')

                            # B_r expression (fixed at 0)
                            Br_expr_oscillatory = sp.sympify(a_Br_rounded_oscillatory)  # This will be 0

                            # B_y expression
                            By_expr_oscillatory = (sp.sympify(A_By_rounded_oscillatory) * r**2 * (
                                sp.sympify(B_By_rounded_oscillatory) +
                                sp.sympify(C_By_rounded_oscillatory) * sp.sin(alpha - sp.sympify(alpha0_By_rounded_oscillatory)) +
                                sp.sympify(D_By_rounded_oscillatory) * sp.cos(alpha - sp.sympify(alpha0_By_rounded_oscillatory))
                            )) + sp.sympify(E_By_rounded_oscillatory)

                            # B_phi expression
                            Bphi_expr_oscillatory = (sp.sympify(A1_Bphi_rounded_oscillatory) * r +
                                                    sp.sympify(B1_Bphi_rounded_oscillatory) * r**2 +
                                                    sp.sympify(C1_Bphi_rounded_oscillatory) * r**3)

                            # SAVING VARIABLES:
                            best_combination_oscillatory = (z0, angle_x, angle_y, angle_z, delta)

                            quality_factors_oscillatory = (R2_B_oscillatory, R2_Bx_oscillatory, R2_By_oscillatory, R2_Bz_oscillatory, R2_avg_oscillatory)
                            # Plot 1: Vertical cut in the cylinder:
                            plot1_vars_oscillatory = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs)

                            # Plot 2: Projection and Rotation in Local Coordinates
                            plot2_vars_oscillatory = (x_local, y_local, x_ellipse_local, y_ellipse_local, x_inter_local, y_inter_local, x_ellipse_rotated, y_ellipse_rotated, x_traj_rotated, y_traj_rotated, x_inter_rotated, y_inter_rotated)

                            # Plot 3: 3D Representation
                            plot3_vars_oscillatory = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)

                            # Plot 4: Radial distance and angles in the trajectory (the 4 plots)
                            plot4_vars_oscillatory = (r_vals, phi_vals, r_elliptic, phi_elliptic, original_x_traj, alpha_traj, r_vals_local_eliptic_oscillatory, alpha_traj_elliptic_oscillatory)

                            # Plot 5: In-situ data in Local Cartesian coordinates
                            plot5_vars_oscillatory = (Bx_local_exp, By_local_exp, Bz_local_exp, B_total_local_exp)

                            # Plot 6: In-situ Cylindrical Components with fitting
                            plot6_vars_oscillatory = (Br_exp, By_exp, Bphi_exp, B_total_exp_cyl, Br_vector_oscillatory, By_vector_oscillatory, Bphi_vector_oscillatory, B_vector_oscillatory)

                            # Plot 7: Fitted local and GSE components
                            plot7_vars_oscillatory = (x_traj_oscillatory, B_vector_oscillatory, Bx_traj_oscillatory, By_traj_cartesian_oscillatory, Bz_traj_oscillatory, Bx_GSE_oscillatory, By_GSE_oscillatory, Bz_GSE_oscillatory)
                            
                            viz_3d_vars_opt_oscillatory = (X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled,
                                                        scale_factor, a, Z_max_scaled, z_cut_scaled, x1_scaled, x2_scaled, X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 
                                                        X_proj_inter, Y_proj_inter, Z_proj_inter, X_proj_traj, Y_proj_traj, Z_proj_traj, d, axis_cylinder_norm)


    progress_bar.progress(100)
    progress_text.text("Processing complete! ✅")


    # POST PROCESSING
    ### (A). Data extraction
    st.success(f"Best R2 general found: {best_R2_general:.4f}")
    st.success(f"Best R2 radial found: {best_R2_radial:.4f}")
    st.success(f"Best R2 oscillatory found: {best_R2_oscillatory:.4f}")
    
    # Unpacking variables for the General Model
    z0_general, angle_x_general, angle_y_general, angle_z_general, delta_general = best_combination_general
    ddoy_data_general, B_data_general, Bx_data_general, By_data_general, Bz_data_general = data_tuple
    R2_B_general, R2_Bx_general, R2_By_general, R2_Bz_general, R2_avg_general = quality_factors_general

    X_rot_scaled_general, Y_rot_scaled_general, Z_rot_scaled_general, X_ellipse_scaled_general, Z_ellipse_scaled_general, X_intersections_scaled_general, Z_intersections_scaled_general, percentage_in_upper_half_general, z_cut_scaled_general, xs_general, ys_general, zs_general = plot1_vars_general
    x_local_general, y_local_general, x_ellipse_local_general, y_ellipse_local_general, x_inter_local_general, y_inter_local_general, x_ellipse_rotated_general, y_ellipse_rotated_general, x_traj_rotated_general, y_traj_rotated_general, x_inter_rotated_general, y_inter_rotated_general = plot2_vars_general
    X_rot_general, Y_rot_general, Z_rot_general, X_ellipse_scaled_general_plot3, Z_ellipse_scaled_general_plot3, X_intersections_scaled_general_plot3, Z_intersections_scaled_general_plot3, x1_scaled_general, x2_scaled_general, X_proj_ellipse_general, Y_proj_ellipse_general, Z_proj_ellipse_general, X_proj_inter_general, Y_proj_inter_general, Z_proj_inter_general, X_proj_traj_general, Y_proj_traj_general, Z_proj_traj_general, d_general, axis_cylinder_norm_general = plot3_vars_general
    r_vals_general, phi_vals_general, r_elliptic_general, phi_elliptic_general, original_x_traj_general, alpha_traj_general, r_vals_local_eliptic_general, alpha_traj_elliptic_general = plot4_vars_general
    Bx_local_exp_general, By_local_exp_general, Bz_local_exp_general, B_total_local_exp_general = plot5_vars_general
    Br_exp_general, By_exp_general, Bphi_exp_general, B_total_exp_cyl_general, Br_vector_general, By_vector_general, Bphi_vector_general, B_vector_general = plot6_vars_general
    x_traj_general, B_vector_general_plot7, Bx_traj_general, By_traj_cartesian_general, Bz_traj_general, Bx_GSE_general, By_GSE_general, Bz_GSE_general = plot7_vars_general

    A1_Br_general, B1_Br_general, C1_Br_general, D1_Br_general, E1_Br_general, alpha0_Br_general = params_Br_fit_general
    A_By_general, B_By_general, C_By_general, D_By_general, alpha0_By_general, E_By_general = params_By_fit_general
    A1_Bphi_general, B1_Bphi_general, C1_Bphi_general, D1_Bphi_general, E1_Bphi_general, alpha0_Bphi_general = params_Bphi_fit_general


    # Unpacking variables for the Radial Model
    z0_radial, angle_x_radial, angle_y_radial, angle_z_radial, delta_radial = best_combination_radial
    ddoy_data_radial, B_data_radial, Bx_data_radial, By_data_radial, Bz_data_radial = data_tuple
    R2_B_radial, R2_Bx_radial, R2_By_radial, R2_Bz_radial, R2_avg_radial = quality_factors_radial

    X_rot_scaled_radial, Y_rot_scaled_radial, Z_rot_scaled_radial, X_ellipse_scaled_radial, Z_ellipse_scaled_radial, X_intersections_scaled_radial, Z_intersections_scaled_radial, percentage_in_upper_half_radial, z_cut_scaled_radial, xs_radial, ys_radial, zs_radial = plot1_vars_radial
    x_local_radial, y_local_radial, x_ellipse_local_radial, y_ellipse_local_radial, x_inter_local_radial, y_inter_local_radial, x_ellipse_rotated_radial, y_ellipse_rotated_radial, x_traj_rotated_radial, y_traj_rotated_radial, x_inter_rotated_radial, y_inter_rotated_radial = plot2_vars_radial
    X_rot_radial, Y_rot_radial, Z_rot_radial, X_ellipse_scaled_radial_plot3, Z_ellipse_scaled_radial_plot3, X_intersections_scaled_radial_plot3, Z_intersections_scaled_radial_plot3, x1_scaled_radial, x2_scaled_radial, X_proj_ellipse_radial, Y_proj_ellipse_radial, Z_proj_ellipse_radial, X_proj_inter_radial, Y_proj_inter_radial, Z_proj_inter_radial, X_proj_traj_radial, Y_proj_traj_radial, Z_proj_traj_radial, d_radial, axis_cylinder_norm_radial = plot3_vars_radial
    r_vals_radial, phi_vals_radial, r_elliptic_radial, phi_elliptic_radial, original_x_traj_radial, alpha_traj_radial, r_vals_local_eliptic_radial, alpha_traj_elliptic_radial = plot4_vars_radial
    Bx_local_exp_radial, By_local_exp_radial, Bz_local_exp_radial, B_total_local_exp_radial = plot5_vars_radial
    Br_exp_radial, By_exp_radial, Bphi_exp_radial, B_total_exp_cyl_radial, Br_vector_radial, By_vector_radial, Bphi_vector_radial, B_vector_radial = plot6_vars_radial
    x_traj_radial, B_vector_radial_plot7, Bx_traj_radial, By_traj_cartesian_radial, Bz_traj_radial, Bx_GSE_radial, By_GSE_radial, Bz_GSE_radial = plot7_vars_radial

    a_Br_radial, = params_Br_fit_radial
    A_By_radial, B_By_radial = params_By_fit_radial
    C_Bphi_radial, = params_Bphi_fit_radial


    # Unpacking variables for the Oscillatory Model
    z0_oscillatory, angle_x_oscillatory, angle_y_oscillatory, angle_z_oscillatory, delta_oscillatory = best_combination_oscillatory
    ddoy_data_oscillatory, B_data_oscillatory, Bx_data_oscillatory, By_data_oscillatory, Bz_data_oscillatory = data_tuple
    R2_B_oscillatory, R2_Bx_oscillatory, R2_By_oscillatory, R2_Bz_oscillatory, R2_avg_oscillatory = quality_factors_oscillatory

    X_rot_scaled_oscillatory, Y_rot_scaled_oscillatory, Z_rot_scaled_oscillatory, X_ellipse_scaled_oscillatory, Z_ellipse_scaled_oscillatory, X_intersections_scaled_oscillatory, Z_intersections_scaled_oscillatory, percentage_in_upper_half_oscillatory, z_cut_scaled_oscillatory, xs_oscillatory, ys_oscillatory, zs_oscillatory = plot1_vars_oscillatory
    x_local_oscillatory, y_local_oscillatory, x_ellipse_local_oscillatory, y_ellipse_local_oscillatory, x_inter_local_oscillatory, y_inter_local_oscillatory, x_ellipse_rotated_oscillatory, y_ellipse_rotated_oscillatory, x_traj_rotated_oscillatory, y_traj_rotated_oscillatory, x_inter_rotated_oscillatory, y_inter_rotated_oscillatory = plot2_vars_oscillatory
    X_rot_oscillatory, Y_rot_oscillatory, Z_rot_oscillatory, X_ellipse_scaled_oscillatory_plot3, Z_ellipse_scaled_oscillatory_plot3, X_intersections_scaled_oscillatory_plot3, Z_intersections_scaled_oscillatory_plot3, x1_scaled_oscillatory, x2_scaled_oscillatory, X_proj_ellipse_oscillatory, Y_proj_ellipse_oscillatory, Z_proj_ellipse_oscillatory, X_proj_inter_oscillatory, Y_proj_inter_oscillatory, Z_proj_inter_oscillatory, X_proj_traj_oscillatory, Y_proj_traj_oscillatory, Z_proj_traj_oscillatory, d_oscillatory, axis_cylinder_norm_oscillatory = plot3_vars_oscillatory
    r_vals_oscillatory, phi_vals_oscillatory, r_elliptic_oscillatory, phi_elliptic_oscillatory, original_x_traj_oscillatory, alpha_traj_oscillatory, r_vals_local_eliptic_oscillatory, alpha_traj_elliptic_oscillatory = plot4_vars_oscillatory
    Bx_local_exp_oscillatory, By_local_exp_oscillatory, Bz_local_exp_oscillatory, B_total_local_exp_oscillatory = plot5_vars_oscillatory
    Br_exp_oscillatory, By_exp_oscillatory, Bphi_exp_oscillatory, B_total_exp_cyl_oscillatory, Br_vector_oscillatory, By_vector_oscillatory, Bphi_vector_oscillatory, B_vector_oscillatory = plot6_vars_oscillatory
    x_traj_oscillatory, B_vector_oscillatory_plot7, Bx_traj_oscillatory, By_traj_cartesian_oscillatory, Bz_traj_oscillatory, Bx_GSE_oscillatory, By_GSE_oscillatory, Bz_GSE_oscillatory = plot7_vars_oscillatory

    a_Br_oscillatory, = params_Br_fit_oscillatory
    A_By_oscillatory, B_By_oscillatory, C_By_oscillatory, D_By_oscillatory, alpha0_By_oscillatory, E_By_oscillatory = params_By_fit_oscillatory
    A1_Bphi_oscillatory, B1_Bphi_oscillatory, C1_Bphi_oscillatory = params_Bphi_fit_oscillatory



    # ----------------------------------------------------------------------------------
    # Plot 01) Fitting to the original GSE data (Radial Model)
    # ----------------------------------------------------------------------------------
    B_vector_radial = B_vector_radial[::-1]
    Bx_GSE_radial = Bx_GSE_radial[::-1]
    By_GSE_radial = By_GSE_radial[::-1]
    Bz_GSE_radial = Bz_GSE_radial[::-1]
    adjusted_data_radial = [B_vector_radial, Bx_GSE_radial, By_GSE_radial, Bz_GSE_radial]

    # --- Gráfico comparativo en Streamlit ---
    st.subheader("1) Model 1 Results (Radial Model)")


    # Fitted formulas of the magnetic field components (Radial Model)
    # ----------------------------------------------------------------------------------
    st.markdown("<h2 style='font-size:20px;'>1.1) General Form of Magnetic Field Components for Radial Model</h2>", unsafe_allow_html=True)
    st.latex(r"B_r(r,\varphi) = 0")
    st.latex(r"B_y(r,\varphi) = A + B r^2")
    st.latex(r"B_\phi(r,\varphi) = C r")

    st.markdown("<h2 style='font-size:20px;'>1.2) Resulting Formulas (Radial Model)</h2>", unsafe_allow_html=True)
    st.latex(f"B_r = {sp.latex(Br_expr_radial)}")
    st.latex(f"B_y = {sp.latex(By_expr_radial)}")
    st.latex(f"B_\\phi = {sp.latex(Bphi_expr_radial)}")
    st.latex(r"\nabla \cdot \mathbf{B} = 0")



    st.markdown("<h2 style='font-size:20px;'>1.3) Fitted to in-situ data</h2>", unsafe_allow_html=True)

    fig_compare_radial, ax_radial = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    components_radial = ['B', 'Bx', 'By', 'Bz']
    data_compare_radial = [B_data_radial, Bx_data_radial, By_data_radial, Bz_data_radial]  # Datos originales en GSE
    titles_compare_radial = [
        "Magnetic Field Intensity (B)",
        "Magnetic Field Component Bx",
        "Magnetic Field Component By",
        "Magnetic Field Component Bz",
    ]

    # Definir los puntos de inicio y fin del segmento
    start_segment_radial = ddoy_data_radial[initial_point - 1]
    end_segment_radial = ddoy_data_radial[final_point - 2]

    for i, (component, orig_data, adj_data, title) in enumerate(
        zip(components_radial, data_compare_radial, adjusted_data_radial, titles_compare_radial)
    ):
        # Plot datos experimentales como puntos negros
        ax_radial[i].scatter(ddoy_data_radial, orig_data, color='black', s=10, label=f'{component} Original')
        
        # Plot datos ajustados como línea roja discontinua (si están disponibles)
        if adj_data is not None:
            ax_radial[i].plot(ddoy_data_radial[initial_point - 1:final_point - 1], adj_data, 
                            color='red', linestyle='--', linewidth=2, label=f'{component} Fitted')
        
        # Líneas verticales para el inicio y fin del segmento
        ax_radial[i].axvline(x=start_segment_radial, color='gray', linestyle='--', label='Start of Segment')
        ax_radial[i].axvline(x=end_segment_radial, color='gray', linestyle='--', label='End of Segment')
        
        # Configuración del subplot
        ax_radial[i].set_title(title, fontsize=18, fontweight='bold')
        ax_radial[i].set_ylabel(f"{component} (nT)", fontsize=14)
        ax_radial[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_radial[i].minorticks_on()
        ax_radial[i].legend(fontsize=12)

    ax_radial[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_compare_radial)
    plt.close(fig_compare_radial)

    # --- Resultados de los parámetros de ajuste ---
    st.markdown("<h2 style='font-size:20px;'>1.4) Fitting Parameters (Radial Model)</h2>", unsafe_allow_html=True)

    # Asumiendo que z0_radial, angle_x_radial, etc., están definidos tras unpacking
    st.latex(f"z_0 = {z0_radial:.2f}")
    st.latex(f"\\theta_x = {np.rad2deg(angle_x_radial):.2f}^\\circ")
    st.latex(f"\\theta_y = {np.rad2deg(angle_y_radial):.2f}^\\circ")
    st.latex(f"\\theta_z = {np.rad2deg(angle_z_radial):.2f}^\\circ")
    st.latex(f"\\delta = {delta_radial:.2f}")



    # # Mostrar los resultados en Streamlit
    # st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results (Radial Model)</h2>", unsafe_allow_html=True)
    # st.latex(f"R^2_B = {R2_B_radial:.4f}")
    # st.latex(f"R^2_{{Bx}} = {R2_Bx_radial:.4f}")
    # st.latex(f"R^2_{{By}} = {R2_By_radial:.4f}")
    # st.latex(f"R^2_{{Bz}} = {R2_Bz_radial:.4f}")
    # st.latex(f"R^2_{{avg}} = {R2_avg_radial:.4f}")








    # Plot 02) Fitting to the original GSE data (Oscillatory Model)
    # ----------------------------------------------------------------------------------
    B_vector_oscillatory = B_vector_oscillatory[::-1]
    Bx_GSE_oscillatory = Bx_GSE_oscillatory[::-1]
    By_GSE_oscillatory = By_GSE_oscillatory[::-1]
    Bz_GSE_oscillatory = Bz_GSE_oscillatory[::-1]
    adjusted_data_oscillatory = [B_vector_oscillatory, Bx_GSE_oscillatory, By_GSE_oscillatory, Bz_GSE_oscillatory]

    # --- Gráfico comparativo en Streamlit ---
    st.subheader("2) Model 2 Results (Radial-Angular Model)")

    # Fitted formulas of the magnetic field components (Oscillatory Model)
    # ----------------------------------------------------------------------------------
    st.markdown("<h2 style='font-size:20px;'>2.1) General Form of Magnetic Field Components</h2>", unsafe_allow_html=True)
    st.latex(r"B_r(r,\varphi) = 0")
    st.latex(r"B_y(r,\varphi) = A r^2 \left[ B + C \sin(\alpha - \alpha_0) + D \cos(\alpha - \alpha_0) \right] + E")
    st.latex(r"B_\phi(r,\varphi) = A_{1} r + B_{1} r^2 + C_{1} r^3")

    st.markdown("<h2 style='font-size:20px;'>2.2) Resulting Formulas </h2>", unsafe_allow_html=True)
    st.latex(f"B_r = {sp.latex(Br_expr_oscillatory)}")
    st.latex(f"B_y = {sp.latex(By_expr_oscillatory)}")
    st.latex(f"B_\\phi = {sp.latex(Bphi_expr_oscillatory)}")
    # st.latex(r"\nabla \cdot \mathbf{B} = 0")

    st.markdown("<h2 style='font-size:20px;'>2.3) Fitted to in-situ data</h2>", unsafe_allow_html=True)

    fig_compare_oscillatory, ax_oscillatory = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    components_oscillatory = ['B', 'Bx', 'By', 'Bz']
    data_compare_oscillatory = [B_data_oscillatory, Bx_data_oscillatory, By_data_oscillatory, Bz_data_oscillatory]  # Datos originales en GSE
    titles_compare_oscillatory = [
        "Magnetic Field Intensity (B)",
        "Magnetic Field Component Bx",
        "Magnetic Field Component By",
        "Magnetic Field Component Bz",
    ]

    # Definir los puntos de inicio y fin del segmento
    start_segment_oscillatory = ddoy_data_oscillatory[initial_point - 1]
    end_segment_oscillatory = ddoy_data_oscillatory[final_point - 2]

    for i, (component, orig_data, adj_data, title) in enumerate(
        zip(components_oscillatory, data_compare_oscillatory, adjusted_data_oscillatory, titles_compare_oscillatory)
    ):
        # Plot datos experimentales como puntos negros
        ax_oscillatory[i].scatter(ddoy_data_oscillatory, orig_data, color='black', s=10, label=f'{component} Original')
        
        # Plot datos ajustados como línea roja discontinua (si están disponibles)
        if adj_data is not None:
            ax_oscillatory[i].plot(ddoy_data_oscillatory[initial_point - 1:final_point - 1], adj_data, 
                                color='red', linestyle='--', linewidth=2, label=f'{component} Fitted')
        
        # Líneas verticales para el inicio y fin del segmento
        ax_oscillatory[i].axvline(x=start_segment_oscillatory, color='gray', linestyle='--', label='Start of Segment')
        ax_oscillatory[i].axvline(x=end_segment_oscillatory, color='gray', linestyle='--', label='End of Segment')
        
        # Configuración del subplot
        ax_oscillatory[i].set_title(title, fontsize=18, fontweight='bold')
        ax_oscillatory[i].set_ylabel(f"{component} (nT)", fontsize=14)
        ax_oscillatory[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_oscillatory[i].minorticks_on()
        ax_oscillatory[i].legend(fontsize=12)

    ax_oscillatory[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_compare_oscillatory)
    plt.close(fig_compare_oscillatory)

    # --- Resultados de los parámetros de ajuste ---
    st.markdown("<h2 style='font-size:20px;'>2.4) Fitting Parameters</h2>", unsafe_allow_html=True)

    # Asumiendo que z0_oscillatory, angle_x_oscillatory, etc., están definidos tras unpacking
    st.latex(f"z_0 = {z0_oscillatory:.2f}")
    st.latex(f"\\theta_x = {np.rad2deg(angle_x_oscillatory):.2f}^\\circ")
    st.latex(f"\\theta_y = {np.rad2deg(angle_y_oscillatory):.2f}^\\circ")
    st.latex(f"\\theta_z = {np.rad2deg(angle_z_oscillatory):.2f}^\\circ")
    st.latex(f"\\delta = {delta_oscillatory:.2f}")

    # # Mostrar los resultados en Streamlit
    # st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results (Oscillatory Model)</h2>", unsafe_allow_html=True)
    # st.latex(f"R^2_B = {R2_B_oscillatory:.4f}")
    # st.latex(f"R^2_{{Bx}} = {R2_Bx_oscillatory:.4f}")
    # st.latex(f"R^2_{{By}} = {R2_By_oscillatory:.4f}")
    # st.latex(f"R^2_{{Bz}} = {R2_Bz_oscillatory:.4f}")
    # st.latex(f"R^2_{{avg}} = {R2_avg_oscillatory:.4f}")








    # ----------------------------------------------------------------------------------
    # Plot 03) Fitting to the original GSE data (General Model)
    # ----------------------------------------------------------------------------------
    B_vector_general = B_vector_general[::-1]
    Bx_GSE_general = Bx_GSE_general[::-1]
    By_GSE_general = By_GSE_general[::-1]
    Bz_GSE_general = Bz_GSE_general[::-1]
    adjusted_data_general = [B_vector_general, Bx_GSE_general, By_GSE_general, Bz_GSE_general]

    # --- Gráfico comparativo en Streamlit ---
    st.subheader("3) Model 3 Results - General Form")

    # Fitted formulas of the magnetic field components (General Model)
    # ----------------------------------------------------------------------------------
    st.markdown("<h2 style='font-size:20px;'>3.1) General Form of Magnetic Field Components for General Model</h2>", unsafe_allow_html=True)
    st.latex(r"B_r(r,\varphi) = A_{1r} r + B_{1r} r^2 \left[ C_{1r} + D_{1r} \sin(\alpha - \alpha_{0r}) + E_{1r} \cos(\alpha - \alpha_{0r}) \right]")
    st.latex(r"B_y(r,\varphi) = A_y r^2 \left[ B_y + C_y \sin(\alpha - \alpha_{0y}) + D_y \cos(\alpha - \alpha_{0y}) \right] + E_y")
    st.latex(r"B_\phi(r,\varphi) = A_{1\phi} r + B_{1\phi} r^2 \left[ C_{1\phi} + D_{1\phi} \sin(\alpha - \alpha_{0\phi}) + E_{1\phi} \cos(\alpha - \alpha_{0\phi}) \right]")

    st.markdown("<h2 style='font-size:20px;'>3.2) Resulting Formulas (General Model)</h2>", unsafe_allow_html=True)
    st.latex(f"B_r = {sp.latex(Br_expr_general)}")
    st.latex(f"B_y = {sp.latex(By_expr_general)}")
    st.latex(f"B_\\phi = {sp.latex(Bphi_expr_general)}")
    st.latex(r"\nabla \cdot \mathbf{B} = 0")

    st.markdown("<h2 style='font-size:20px;'>3.3) Fitted to in-situ data</h2>", unsafe_allow_html=True)

    fig_compare_general, ax_general = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    components_general = ['B', 'Bx', 'By', 'Bz']
    data_compare_general = [B_data_general, Bx_data_general, By_data_general, Bz_data_general]  # Datos originales en GSE
    titles_compare_general = [
        "Magnetic Field Intensity (B)",
        "Magnetic Field Component Bx",
        "Magnetic Field Component By",
        "Magnetic Field Component Bz",
    ]

    # Definir los puntos de inicio y fin del segmento
    start_segment_general = ddoy_data_general[initial_point - 1]
    end_segment_general = ddoy_data_general[final_point - 2]

    for i, (component, orig_data, adj_data, title) in enumerate(
        zip(components_general, data_compare_general, adjusted_data_general, titles_compare_general)
    ):
        # Plot datos experimentales como puntos negros
        ax_general[i].scatter(ddoy_data_general, orig_data, color='black', s=10, label=f'{component} Original')
        
        # Plot datos ajustados como línea roja discontinua (si están disponibles)
        if adj_data is not None:
            ax_general[i].plot(ddoy_data_general[initial_point - 1:final_point - 1], adj_data, 
                            color='red', linestyle='--', linewidth=2, label=f'{component} Fitted')
        
        # Líneas verticales para el inicio y fin del segmento
        ax_general[i].axvline(x=start_segment_general, color='gray', linestyle='--', label='Start of Segment')
        ax_general[i].axvline(x=end_segment_general, color='gray', linestyle='--', label='End of Segment')
        
        # Configuración del subplot
        ax_general[i].set_title(title, fontsize=18, fontweight='bold')
        ax_general[i].set_ylabel(f"{component} (nT)", fontsize=14)
        ax_general[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_general[i].minorticks_on()
        ax_general[i].legend(fontsize=12)

    ax_general[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_compare_general)
    plt.close(fig_compare_general)

    # --- Resultados de los parámetros de ajuste ---
    st.markdown("<h2 style='font-size:20px;'>Fitting Parameters (General Model)</h2>", unsafe_allow_html=True)

    # Asumiendo que z0_general, angle_x_general, etc., están definidos tras unpacking
    st.latex(f"z_0 = {z0_general:.2f}")
    st.latex(f"\\theta_x = {np.rad2deg(angle_x_general):.2f}^\\circ")
    st.latex(f"\\theta_y = {np.rad2deg(angle_y_general):.2f}^\\circ")
    st.latex(f"\\theta_z = {np.rad2deg(angle_z_general):.2f}^\\circ")
    st.latex(f"\\delta = {delta_general:.2f}")

    # # Mostrar los resultados en Streamlit
    # st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results (General Model)</h2>", unsafe_allow_html=True)
    # st.latex(f"R^2_B = {R2_B_general:.4f}")
    # st.latex(f"R^2_{{Bx}} = {R2_Bx_general:.4f}")
    # st.latex(f"R^2_{{By}} = {R2_By_general:.4f}")
    # st.latex(f"R^2_{{Bz}} = {R2_Bz_general:.4f}")
    # st.latex(f"R^2_{{avg}} = {R2_avg_general:.4f}")


    # Display the consolidated R² table
    r2_data = {
        "Component": ["R2_B", "R2_Bx", "R2_By", "R2_Bz", "R2_Average"],
        "General Model": [R2_B_general, R2_Bx_general, R2_By_general, R2_Bz_general, R2_avg_general],
        "Radial Model": [R2_B_radial, R2_Bx_radial, R2_By_radial, R2_Bz_radial, R2_avg_radial],
        "Oscillatory Model": [R2_B_oscillatory, R2_Bx_oscillatory, R2_By_oscillatory, R2_Bz_oscillatory, R2_avg_oscillatory]
    }
    r2_df = pd.DataFrame(r2_data)
    r2_df["General Model"] = r2_df["General Model"].map("{:.4f}".format)
    r2_df["Radial Model"] = r2_df["Radial Model"].map("{:.4f}".format)
    r2_df["Oscillatory Model"] = r2_df["Oscillatory Model"].map("{:.4f}".format)

    st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results Across Models</h2>", unsafe_allow_html=True)
    st.dataframe(r2_df, use_container_width=True)




    # Crear un diccionario con los parámetros de ajuste para cada modelo
    fitting_params_data = {
        "Parameter": [
            r"z0",
            r"theta_x (°)",
            r"theta_y (°)",
            r"theta_z (°)",
            r"delta"
        ],
        "General Model": [
            z0_general,
            np.rad2deg(angle_x_general),
            np.rad2deg(angle_y_general),
            np.rad2deg(angle_z_general),
            delta_general
        ],
        "Radial Model": [
            z0_radial,
            np.rad2deg(angle_x_radial),
            np.rad2deg(angle_y_radial),
            np.rad2deg(angle_z_radial),
            delta_radial
        ],
        "Oscillatory Model": [
            z0_oscillatory,
            np.rad2deg(angle_x_oscillatory),
            np.rad2deg(angle_y_oscillatory),
            np.rad2deg(angle_z_oscillatory),
            delta_oscillatory
        ]
    }

    # Convertir el diccionario a DataFrame
    fitting_params_df = pd.DataFrame(fitting_params_data)

    # Formatear las columnas numéricas a 2 decimales
    fitting_params_df["General Model"] = fitting_params_df["General Model"].map("{:.2f}".format)
    fitting_params_df["Radial Model"] = fitting_params_df["Radial Model"].map("{:.2f}".format)
    fitting_params_df["Oscillatory Model"] = fitting_params_df["Oscillatory Model"].map("{:.2f}".format)

    # Usar la columna "Parameter" como índice
    fitting_params_df.set_index("Parameter", inplace=True)

    # Mostrar la tabla en Streamlit
    st.markdown("<h2 style='font-size:20px;'>Fitting Parameters Across Models</h2>", unsafe_allow_html=True)
    st.dataframe(fitting_params_df, use_container_width=True)



    # Compare R²_avg values to find the model with the highest value
    r2_values = {
        "General": R2_avg_general,
        "Radial": R2_avg_radial,
        "Oscillatory": R2_avg_oscillatory
    }

    # Find the model with the highest R²_avg
    best_model = max(r2_values, key=r2_values.get)
    best_r2 = r2_values[best_model]

    # Select the corresponding viz_3d_vars_opt based on the best model
    if best_model == "General":
        viz_3d_vars_opt_best = viz_3d_vars_opt_general
    elif best_model == "Radial":
        viz_3d_vars_opt_best = viz_3d_vars_opt_radial
    else:  # Oscillatory
        viz_3d_vars_opt_best = viz_3d_vars_opt_oscillatory

    st.markdown(f"<h2 style='font-size:22px;'>3D Visualization for Best Model ({best_model}, R²_avg = {best_r2:.4f})</h2>", unsafe_allow_html=True)

    # Call trajectory_3d with the best model's variables
    trajectory_3d(viz_3d_vars_opt_best)



















    # st.subheader("1) Geometry of the fitted Flux Rope (General Model)")
    # # ----------------------------------------------------------------------------------
    # # Plot 1) Vertical cut in the cylinder
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.1) Vertical cut in the cylinder (General Model)</h2>", unsafe_allow_html=True)

    # x_min_general = np.min(X_ellipse_scaled_general)
    # x_max_general = np.max(X_ellipse_scaled_general)
    # z_min_general = np.min(Z_ellipse_scaled_general)
    # z_max_general = np.max(Z_ellipse_scaled_general)
    # margin_general = 0.1 * (x_max_general - x_min_general)
    # x_limits_general = [x_min_general - margin_general, x_max_general + margin_general]
    # z_limits_general = [z_min_general - margin_general, z_max_general + margin_general]

    # max_z_index_general = np.argmax(Z_ellipse_scaled_general)
    # x_max_z_general = X_ellipse_scaled_general[max_z_index_general]

    # indices_z_zero_general = np.where(np.abs(Z_ellipse_scaled_general - 0) < 1e-2)[0]
    # x_z_zero_limits_general = np.sort(X_ellipse_scaled_general[indices_z_zero_general]) if len(indices_z_zero_general) >= 2 else [x_min_general, x_max_general]
    # indices_x_max_z_general = np.where(np.abs(X_ellipse_scaled_general - x_max_z_general) < 1e-2)[0]
    # z_x_max_z_limits_general = np.sort(Z_ellipse_scaled_general[indices_x_max_z_general]) if len(indices_x_max_z_general) >= 2 else [z_min_general, z_max_general]

    # z_mid_general = (z_x_max_z_limits_general[0] + z_x_max_z_limits_general[1]) / 2
    # z_upper_limit_general = z_x_max_z_limits_general[1]
    # z_lower_limit_upper_half_general = z_mid_general
    # height_upper_half_general = z_upper_limit_general - z_lower_limit_upper_half_general
    # relative_position_general = z_cut_scaled_general - z_lower_limit_upper_half_general
    # percentage_in_upper_half_general = (relative_position_general / height_upper_half_general) * 100 if height_upper_half_general > 0 else 0

    # # st.write(f"Percentage of the trajectory height in the upper half of maximum z (General Model): {percentage_in_upper_half_general:.2f}%")

    # # st.write(f"Percentage of the trajectory height in the upper half of maximum z: {percentage_in_upper_half:.2f}%")
    
    # # ----------------------------------------------------------------------------------
    # # 1A) GENERAL
    # # ----------------------------------------------------------------------------------
    # fig_general = plt.figure(figsize=(16, 8))
    # #                                    3D Plot
    # # ----------------------------------------------------------------------------------
    # ax1_general = fig_general.add_subplot(121, projection='3d')
    # ax1_general.plot_surface(X_rot_scaled_general, Y_rot_scaled_general, Z_rot_scaled_general, color='lightblue', alpha=0.4)
    # ax1_general.plot(X_ellipse_scaled_general, np.zeros_like(X_ellipse_scaled_general), Z_ellipse_scaled_general, 'r', label="Scaled ellipse")
    # ax1_general.scatter(X_intersections_scaled_general, np.zeros_like(X_intersections_scaled_general), Z_intersections_scaled_general, color='red', s=50, label="Scaled intersection")
    # ax1_general.plot(xs_general, ys_general, zs_general, 'g-', linewidth=2, label="Satellite trajectory")
    # y_plane_general = np.zeros((10, 10))
    # x_plane_general = np.linspace(x_limits[0], x_limits[1], 10)
    # z_plane_general = np.linspace(z_limits[0], z_limits[1], 10)
    # X_plane_general, Z_plane_general = np.meshgrid(x_plane_general, z_plane_general)
    # ax1_general.plot_surface(X_plane_general, y_plane_general, Z_plane_general, color='gray', alpha=0.2)
    # ax1_general.plot(x_z_zero_limits, [0, 0], [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax1_general.plot([x_max_z, x_max_z], [0, 0], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax1_general.set_xlabel("X")
    # ax1_general.set_ylabel("Y")
    # ax1_general.set_zlabel("Z")
    # ax1_general.set_title("3D: Centered Cylinder with Trajectory (General Model)")
    # ax1_general.legend()
    # set_axes_equal(ax1_general)

    # #                                   2D Plot
    # # ----------------------------------------------------------------------------------
    # ax2_general = fig_general.add_subplot(122)
    # ax2_general.plot(X_ellipse_scaled_general, Z_ellipse_scaled_general, 'r', label="Scaled ellipse")
    # ax2_general.scatter(X_intersections_scaled_general, Z_intersections_scaled_general, color='red', s=50, label="Scaled intersection")
    # ax2_general.plot(xs_general, zs_general, 'g-', linewidth=2, label="Satellite trajectory")
    # ax2_general.plot(x_z_zero_limits, [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax2_general.plot([x_max_z, x_max_z], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax2_general.set_xlabel("X")
    # ax2_general.set_ylabel("Z")
    # ax2_general.set_title("2D: Centered Elliptical Section (General Model)")
    # ax2_general.set_xlim(x_limits)
    # ax2_general.set_ylim(z_limits)
    # ax2_general.legend()
    # ax2_general.set_aspect('equal')

    # plt.tight_layout()
    # st.pyplot(fig_general)


    # st.subheader("1) Geometry of the fitted Flux Rope (Radial Model)")
    # # ----------------------------------------------------------------------------------
    # # Plot 1) Vertical cut in the cylinder
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.1) Vertical cut in the cylinder (Radial Model)</h2>", unsafe_allow_html=True)

    # x_min_radial = np.min(X_ellipse_scaled_radial)
    # x_max_radial = np.max(X_ellipse_scaled_radial)
    # z_min_radial = np.min(Z_ellipse_scaled_radial)
    # z_max_radial = np.max(Z_ellipse_scaled_radial)
    # margin_radial = 0.1 * (x_max_radial - x_min_radial)
    # x_limits_radial = [x_min_radial - margin_radial, x_max_radial + margin_radial]
    # z_limits_radial = [z_min_radial - margin_radial, z_max_radial + margin_radial]

    # max_z_index_radial = np.argmax(Z_ellipse_scaled_radial)
    # x_max_z_radial = X_ellipse_scaled_radial[max_z_index_radial]

    # indices_z_zero_radial = np.where(np.abs(Z_ellipse_scaled_radial - 0) < 1e-2)[0]
    # x_z_zero_limits_radial = np.sort(X_ellipse_scaled_radial[indices_z_zero_radial]) if len(indices_z_zero_radial) >= 2 else [x_min_radial, x_max_radial]
    # indices_x_max_z_radial = np.where(np.abs(X_ellipse_scaled_radial - x_max_z_radial) < 1e-2)[0]
    # z_x_max_z_limits_radial = np.sort(Z_ellipse_scaled_radial[indices_x_max_z_radial]) if len(indices_x_max_z_radial) >= 2 else [z_min_radial, z_max_radial]

    # z_mid_radial = (z_x_max_z_limits_radial[0] + z_x_max_z_limits_radial[1]) / 2
    # z_upper_limit_radial = z_x_max_z_limits_radial[1]
    # z_lower_limit_upper_half_radial = z_mid_radial
    # height_upper_half_radial = z_upper_limit_radial - z_lower_limit_upper_half_radial
    # relative_position_radial = z_cut_scaled_radial - z_lower_limit_upper_half_radial
    # percentage_in_upper_half_radial = (relative_position_radial / height_upper_half_radial) * 100 if height_upper_half_radial > 0 else 0

    # # st.write(f"Percentage of the trajectory height in the upper half of maximum z (Radial Model): {percentage_in_upper_half_radial:.2f}%")


    # # ----------------------------------------------------------------------------------
    # # 1B) RADIAL
    # # ----------------------------------------------------------------------------------
    # fig_radial = plt.figure(figsize=(16, 8))
    # #                                    3D Plot
    # # ----------------------------------------------------------------------------------
    # ax1_radial = fig_radial.add_subplot(121, projection='3d')
    # ax1_radial.plot_surface(X_rot_scaled_radial, Y_rot_scaled_radial, Z_rot_scaled_radial, color='lightblue', alpha=0.4)
    # ax1_radial.plot(X_ellipse_scaled_radial, np.zeros_like(X_ellipse_scaled_radial), Z_ellipse_scaled_radial, 'r', label="Scaled ellipse")
    # ax1_radial.scatter(X_intersections_scaled_radial, np.zeros_like(X_intersections_scaled_radial), Z_intersections_scaled_radial, color='red', s=50, label="Scaled intersection")
    # ax1_radial.plot(xs_radial, ys_radial, zs_radial, 'g-', linewidth=2, label="Satellite trajectory")
    # y_plane_radial = np.zeros((10, 10))
    # x_plane_radial = np.linspace(x_limits[0], x_limits[1], 10)
    # z_plane_radial = np.linspace(z_limits[0], z_limits[1], 10)
    # X_plane_radial, Z_plane_radial = np.meshgrid(x_plane_radial, z_plane_radial)
    # ax1_radial.plot_surface(X_plane_radial, y_plane_radial, Z_plane_radial, color='gray', alpha=0.2)
    # ax1_radial.plot(x_z_zero_limits, [0, 0], [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax1_radial.plot([x_max_z, x_max_z], [0, 0], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax1_radial.set_xlabel("X")
    # ax1_radial.set_ylabel("Y")
    # ax1_radial.set_zlabel("Z")
    # ax1_radial.set_title("3D: Centered Cylinder with Trajectory (Radial Model)")
    # ax1_radial.legend()
    # set_axes_equal(ax1_radial)

    # #                                   2D Plot
    # # ----------------------------------------------------------------------------------
    # ax2_radial = fig_radial.add_subplot(122)
    # ax2_radial.plot(X_ellipse_scaled_radial, Z_ellipse_scaled_radial, 'r', label="Scaled ellipse")
    # ax2_radial.scatter(X_intersections_scaled_radial, Z_intersections_scaled_radial, color='red', s=50, label="Scaled intersection")
    # ax2_radial.plot(xs_radial, zs_radial, 'g-', linewidth=2, label="Satellite trajectory")
    # ax2_radial.plot(x_z_zero_limits, [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax2_radial.plot([x_max_z, x_max_z], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax2_radial.set_xlabel("X")
    # ax2_radial.set_ylabel("Z")
    # ax2_radial.set_title("2D: Centered Elliptical Section (Radial Model)")
    # ax2_radial.set_xlim(x_limits)
    # ax2_radial.set_ylim(z_limits)
    # ax2_radial.legend()
    # ax2_radial.set_aspect('equal')

    # plt.tight_layout()
    # st.pyplot(fig_radial)


    # fig_oscillatory = plt.figure(figsize=(16, 8))



    # # ----------------------------------------------------------------------------------
    # # 1C) RADIAL

    # st.subheader("1) Geometry of the fitted Flux Rope (Oscillatory Model)")
    # # ----------------------------------------------------------------------------------
    # # Plot 1) Vertical cut in the cylinder
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.1) Vertical cut in the cylinder (Oscillatory Model)</h2>", unsafe_allow_html=True)

    # x_min_oscillatory = np.min(X_ellipse_scaled_oscillatory)
    # x_max_oscillatory = np.max(X_ellipse_scaled_oscillatory)
    # z_min_oscillatory = np.min(Z_ellipse_scaled_oscillatory)
    # z_max_oscillatory = np.max(Z_ellipse_scaled_oscillatory)
    # margin_oscillatory = 0.1 * (x_max_oscillatory - x_min_oscillatory)
    # x_limits_oscillatory = [x_min_oscillatory - margin_oscillatory, x_max_oscillatory + margin_oscillatory]
    # z_limits_oscillatory = [z_min_oscillatory - margin_oscillatory, z_max_oscillatory + margin_oscillatory]

    # max_z_index_oscillatory = np.argmax(Z_ellipse_scaled_oscillatory)
    # x_max_z_oscillatory = X_ellipse_scaled_oscillatory[max_z_index_oscillatory]

    # indices_z_zero_oscillatory = np.where(np.abs(Z_ellipse_scaled_oscillatory - 0) < 1e-2)[0]
    # x_z_zero_limits_oscillatory = np.sort(X_ellipse_scaled_oscillatory[indices_z_zero_oscillatory]) if len(indices_z_zero_oscillatory) >= 2 else [x_min_oscillatory, x_max_oscillatory]
    # indices_x_max_z_oscillatory = np.where(np.abs(X_ellipse_scaled_oscillatory - x_max_z_oscillatory) < 1e-2)[0]
    # z_x_max_z_limits_oscillatory = np.sort(Z_ellipse_scaled_oscillatory[indices_x_max_z_oscillatory]) if len(indices_x_max_z_oscillatory) >= 2 else [z_min_oscillatory, z_max_oscillatory]

    # z_mid_oscillatory = (z_x_max_z_limits_oscillatory[0] + z_x_max_z_limits_oscillatory[1]) / 2
    # z_upper_limit_oscillatory = z_x_max_z_limits_oscillatory[1]
    # z_lower_limit_upper_half_oscillatory = z_mid_oscillatory
    # height_upper_half_oscillatory = z_upper_limit_oscillatory - z_lower_limit_upper_half_oscillatory
    # relative_position_oscillatory = z_cut_scaled_oscillatory - z_lower_limit_upper_half_oscillatory
    # percentage_in_upper_half_oscillatory = (relative_position_oscillatory / height_upper_half_oscillatory) * 100 if height_upper_half_oscillatory > 0 else 0

    # # st.write(f"Percentage of the trajectory height in the upper half of maximum z (Oscillatory Model): {percentage_in_upper_half_oscillatory:.2f}%")

    # # ----------------------------------------------------------------------------------
    # #                                    3D Plot
    # # ----------------------------------------------------------------------------------
    # ax1_oscillatory = fig_oscillatory.add_subplot(121, projection='3d')
    # ax1_oscillatory.plot_surface(X_rot_scaled_oscillatory, Y_rot_scaled_oscillatory, Z_rot_scaled_oscillatory, color='lightblue', alpha=0.4)
    # ax1_oscillatory.plot(X_ellipse_scaled_oscillatory, np.zeros_like(X_ellipse_scaled_oscillatory), Z_ellipse_scaled_oscillatory, 'r', label="Scaled ellipse")
    # ax1_oscillatory.scatter(X_intersections_scaled_oscillatory, np.zeros_like(X_intersections_scaled_oscillatory), Z_intersections_scaled_oscillatory, color='red', s=50, label="Scaled intersection")
    # ax1_oscillatory.plot(xs_oscillatory, ys_oscillatory, zs_oscillatory, 'g-', linewidth=2, label="Satellite trajectory")
    # y_plane_oscillatory = np.zeros((10, 10))
    # x_plane_oscillatory = np.linspace(x_limits[0], x_limits[1], 10)
    # z_plane_oscillatory = np.linspace(z_limits[0], z_limits[1], 10)
    # X_plane_oscillatory, Z_plane_oscillatory = np.meshgrid(x_plane_oscillatory, z_plane_oscillatory)
    # ax1_oscillatory.plot_surface(X_plane_oscillatory, y_plane_oscillatory, Z_plane_oscillatory, color='gray', alpha=0.2)
    # ax1_oscillatory.plot(x_z_zero_limits, [0, 0], [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax1_oscillatory.plot([x_max_z, x_max_z], [0, 0], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax1_oscillatory.set_xlabel("X")
    # ax1_oscillatory.set_ylabel("Y")
    # ax1_oscillatory.set_zlabel("Z")
    # ax1_oscillatory.set_title("3D: Centered Cylinder with Trajectory (Oscillatory Model)")
    # ax1_oscillatory.legend()
    # set_axes_equal(ax1_oscillatory)

    # #                                   2D Plot
    # # ----------------------------------------------------------------------------------
    # ax2_oscillatory = fig_oscillatory.add_subplot(122)
    # ax2_oscillatory.plot(X_ellipse_scaled_oscillatory, Z_ellipse_scaled_oscillatory, 'r', label="Scaled ellipse")
    # ax2_oscillatory.scatter(X_intersections_scaled_oscillatory, Z_intersections_scaled_oscillatory, color='red', s=50, label="Scaled intersection")
    # ax2_oscillatory.plot(xs_oscillatory, zs_oscillatory, 'g-', linewidth=2, label="Satellite trajectory")
    # ax2_oscillatory.plot(x_z_zero_limits, [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax2_oscillatory.plot([x_max_z, x_max_z], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax2_oscillatory.set_xlabel("X")
    # ax2_oscillatory.set_ylabel("Z")
    # ax2_oscillatory.set_title("2D: Centered Elliptical Section (Oscillatory Model)")
    # ax2_oscillatory.set_xlim(x_limits)
    # ax2_oscillatory.set_ylim(z_limits)
    # ax2_oscillatory.legend()
    # ax2_oscillatory.set_aspect('equal')

    # plt.tight_layout()
    # st.pyplot(fig_oscillatory)

















    # fig = plt.figure(figsize=(16, 8))
    # #                                    3D Plot
    # # ----------------------------------------------------------------------------------
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(X_rot_scaled, Y_rot_scaled, Z_rot_scaled, color='lightblue', alpha=0.4)
    # ax1.plot(X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled, 'r', label="Scaled ellipse")
    # ax1.scatter(X_intersections_scaled, np.zeros_like(X_intersections_scaled), Z_intersections_scaled, color='red', s=50, label="Scaled intersection")
    # ax1.plot(xs, ys, zs, 'g-', linewidth=2, label="Satellite trajectory")
    # y_plane = np.zeros((10, 10))
    # x_plane = np.linspace(x_limits[0], x_limits[1], 10)
    # z_plane = np.linspace(z_limits[0], z_limits[1], 10)
    # X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
    # ax1.plot_surface(X_plane, y_plane, Z_plane, color='gray', alpha=0.2)
    # ax1.plot(x_z_zero_limits, [0, 0], [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax1.plot([x_max_z, x_max_z], [0, 0], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax1.set_xlabel("X")
    # ax1.set_ylabel("Y")
    # ax1.set_zlabel("Z")
    # ax1.set_title("3D: Centered Cylinder with Trajectory")
    # ax1.legend()
    # set_axes_equal(ax1)

    # #                                   2D Plot
    # # ----------------------------------------------------------------------------------
    # ax2 = fig.add_subplot(122)
    # ax2.plot(X_ellipse_scaled, Z_ellipse_scaled, 'r', label="Scaled ellipse")
    # ax2.scatter(X_intersections_scaled, Z_intersections_scaled, color='red', s=50, label="Scaled intersection")
    # ax2.plot(xs, zs, 'g-', linewidth=2, label="Satellite trajectory")
    # ax2.plot(x_z_zero_limits, [0, 0], 'k--', linewidth=1.5, label="z = 0")
    # ax2.plot([x_max_z, x_max_z], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
    # ax2.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_title("2D: Centered Elliptical Section")
    # ax2.set_xlim(x_limits)
    # ax2.set_ylim(z_limits)
    # ax2.legend()
    # ax2.set_aspect('equal')

    # plt.tight_layout()
    # st.pyplot(fig)



    # # ----------------------------------------------------------------------------------
    # # Plot 2) Projection and Rotation in Local Coordinates
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.2) Section projected in the transversal plane of the FR</h2>", unsafe_allow_html=True)
    # # Create a figure with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # # Subplot 1: Elliptical Section and Projected Trajectory in Local Coordinates
    # ax1.plot(x_ellipse_local, y_ellipse_local, 'r-', label="Elliptical Section")
    # ax1.plot(x_local, y_local, 'b-', linewidth=2, label="Projected Trajectory")
    # ax1.scatter(x_inter_local, y_inter_local, color='blue', s=50, label="Intersection Points")
    # ax1.set_xlabel("Local X")
    # ax1.set_ylabel("Local Y")
    # ax1.set_title("Elliptical Section in Local Coordinates")
    # ax1.legend()
    # ax1.grid(True)
    # ax1.axis('equal')

    # # Subplot 2: Rotated Ellipse with Horizontal Cut
    # ax2.plot(x_ellipse_rotated, y_ellipse_rotated, 'r-', label="Rotated Ellipse")
    # ax2.plot(x_traj_rotated, y_traj_rotated, 'b-', linewidth=2, label="Horizontal Trajectory")
    # ax2.scatter(x_inter_rotated, y_inter_rotated, color='blue', s=50, label="Intersection Points")
    # ax2.set_xlabel("Rotated Local X")
    # ax2.set_ylabel("Rotated Local Y")
    # ax2.set_title("Rotated Ellipse with Horizontal Cut")
    # ax2.legend()
    # ax2.grid(True)
    # ax2.axis('equal')

    # # Adjust layout and display in Streamlit
    # plt.tight_layout()
    # st.pyplot(fig)
    

    # # Plot 3) 3D Representation
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.3) Global 3D representation </h2>", unsafe_allow_html=True)

    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot cylinder surface
    # ax.plot_surface(X_rot_scaled, Y_rot_scaled, Z_rot_scaled, color='lightblue', alpha=0.4)

    # # Plot rescaled ellipse and intersections
    # ax.plot(X_ellipse_scaled, np.zeros_like(X_ellipse_scaled), Z_ellipse_scaled, 'r', label="Rescaled Ellipse")
    # ax.scatter(X_intersections_scaled, np.zeros_like(X_intersections_scaled), Z_intersections_scaled, color='red', s=50, label="Rescaled Intersection")

    # # Plot cutting plane (Y=0)
    # x_plane = np.linspace(-scale_factor * a * 1.5, scale_factor * a * 1.5, 10)
    # z_plane = np.linspace(-Z_max_scaled * 1.5, Z_max_scaled * 1.5, 10)
    # X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
    # y_plane = np.zeros((10, 10))
    # ax.plot_surface(X_plane, y_plane, Z_plane, color='gray', alpha=0.2)

    # # Plot intersection line and extended dashed line
    # ax.plot([x1_scaled, x2_scaled], [0, 0], [z_cut_scaled, z_cut_scaled], 'b-', linewidth=2, label="Intersection Line")
    # x_extended = np.linspace(-a * 1.0, a * 1.0, 100)
    # z_extended = np.full_like(x_extended, z_cut_scaled)
    # ax.plot(x_extended, np.zeros_like(x_extended), z_extended, 'k--', linewidth=1.5, label="Extended Dashed Line")

    # # Plot projected ellipse, intersections, and trajectory
    # ax.plot(X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse, 'g-', label="Projected Ellipse")
    # ax.scatter(X_proj_inter, Y_proj_inter, Z_proj_inter, color='green', s=50, label="Projected Intersection")
    # ax.plot(X_proj_traj, Y_proj_traj, Z_proj_traj, 'm-', linewidth=2, label="Projected Trajectory")

    # # Plot transverse plane
    # x_range = np.linspace(-scale_factor * a * 1.5, scale_factor * a * 1.5, 10)
    # y_range = np.linspace(-a_rotated * 1.5, a_rotated * 1.5, 10)
    # X_trans, Y_trans = np.meshgrid(x_range, y_range)
    # Z_trans = (d - axis_cylinder_norm[0] * X_trans - axis_cylinder_norm[1] * Y_trans) / axis_cylinder_norm[2]
    # ax.plot_surface(X_trans, Y_trans, Z_trans, color='yellow', alpha=0.2, label="Transverse Plane")

    # # Set labels and title
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Cylinder with Cut, Projection, and Trajectory")
    # ax.legend()

    # # Ensure equal aspect ratio
    # set_axes_equal(ax)

    # # Display in Streamlit
    # st.pyplot(fig)     


    # # Plot 4) Radial distance and angles in the trajectory (the 4 plots)
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>1.4) Radial and angular values of the fitted trajectory</h2>", unsafe_allow_html=True)

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # # Plot 1: r_vals vs X_traj_rotated (cartesian)
    # ax1.plot(X_traj_rotated, r_vals, 'b-', label="Radial distance (cartesian)")
    # ax1.scatter(X_traj_rotated, r_vals, c='blue', s=10, label="Points")
    # ax1.set_xlabel("X_rotated")
    # ax1.set_ylabel("r (radial distance, cartesian)")
    # ax1.set_title("9.2.1 Radial Distance (r_vals) vs X_rotated")
    # ax1.grid(True)
    # ax1.legend()

    # # Plot 2: phi_vals vs X_traj_rotated (cartesian)
    # ax2.plot(X_traj_rotated, phi_vals, 'r-', label="Angle (radians, cartesian)")
    # ax2.scatter(X_traj_rotated, phi_vals, c='red', s=10, label="Points")
    # ax2.set_xlabel("X_rotated")
    # ax2.set_ylabel("phi (radians, 0 to 2pi, cartesian)")
    # ax2.set_title("9.2.2 Angle (phi_vals) vs X_rotated")
    # ax2.grid(True)
    # ax2.legend()

    # # Plot 3: r_elliptic vs X_traj_rotated (elliptic)
    # ax3.plot(X_traj_rotated, r_elliptic, 'g-', label="Radial distance (elliptic)")
    # ax3.scatter(X_traj_rotated, r_elliptic, c='green', s=10, label="Points")
    # ax3.set_xlabel("X_rotated")
    # ax3.set_ylabel("r_elliptic (normalized distance, elliptic)")
    # ax3.set_title("9.2.3 Radial Distance (r_elliptic) vs X_rotated")
    # ax3.grid(True)
    # ax3.legend()

    # # Plot 4: phi_elliptic vs X_traj_rotated (elliptic)
    # ax4.plot(X_traj_rotated, phi_elliptic, 'm-', label="Angle (radians, elliptic)")
    # ax4.scatter(X_traj_rotated, phi_elliptic, c='magenta', s=10, label="Points")
    # ax4.set_xlabel("X_rotated")
    # ax4.set_ylabel("phi_elliptic (radians, 0 to 2pi, elliptic)")
    # ax4.set_title("9.2.4 Angle (phi_elliptic) vs X_rotated")
    # ax4.grid(True)
    # ax4.legend()

    # plt.tight_layout()
    # st.pyplot(fig)

    # # Plot 5) In-situ data in Local Cartesian coordinates
    # # ----------------------------------------------------------------------------------
    
    # # Visualization
    # st.subheader("2) In-situ Data in Local Cartesian Coordinates")
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(X_traj_rotated, B_total_local_exp, 'k-', label=r'$|\mathbf{B}|$', linewidth=2)
    # ax.plot(X_traj_rotated, Bx_local_exp, 'b-', label=r'$B_x$', linewidth=2)
    # ax.plot(X_traj_rotated, By_local_exp, 'r-', label=r'$B_y$', linewidth=2)
    # ax.plot(X_traj_rotated, Bz_local_exp, 'g-', label=r'$B_z$', linewidth=2)
    # ax.set_xlabel("Rotated Local X (km)", fontsize=14)
    # ax.set_ylabel("Magnetic Field (nT)", fontsize=14)
    # ax.set_title("Magnetic Field Components in Local Cartesian Coordinates", fontsize=16)
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend(fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # plt.tight_layout()
    # st.pyplot(fig)
    # plt.close(fig)


    # # Plot 6) In-situ Cylindrical Components with fitting
    # # ----------------------------------------------------------------------------------
    # st.subheader("3) Cylindrical Components (Experimental vs Fitted)")
    # fig, ax = plt.subplots(figsize=(12, 8))

    # # --- Plot: Datos Experimentales (colores más tenues) ---
    # ax.plot(original_x_traj, B_total_exp_cyl, color='#808080', linestyle='-', linewidth=2.5, label=r'$|\mathbf{B}|_{exp}$')  # Gris suave
    # ax.plot(original_x_traj, Br_exp, color='#87CEEB', linestyle='-', linewidth=2.5, label=r'$B^r_{exp}$')  # Azul cielo
    # ax.plot(original_x_traj, By_exp, color='#FFA07A', linestyle='-', linewidth=2.5, label=r'$B^y_{exp}$')  # Salmón claro
    # ax.plot(original_x_traj, Bphi_exp, color='#90EE90', linestyle='-', linewidth=2.5, label=r'$B^\varphi_{exp}$')  # Verde claro

    # # --- Plot: Datos Ajustados (colores originales más fuertes) ---
    # ax.plot(original_x_traj, B_vector, 'k--', linewidth=2.5, label=r'$|\mathbf{B}|_{fit}$')  # Negro
    # ax.plot(original_x_traj, Br_vector, 'b--', linewidth=2.5, label=r'$B^r_{fit}$')  # Azul
    # ax.plot(original_x_traj, By_vector, 'r--', linewidth=2.5, label=r'$B^y_{fit}$')  # Rojo
    # ax.plot(original_x_traj, Bphi_vector, 'g--', linewidth=2.5, label=r'$B^\varphi_{fit}$')  # Verde

    # # Configuración del gráfico
    # ax.set_xlabel('X local rotated')
    # ax.set_ylabel('Magnetic Field')
    # ax.set_title('Cylindrical Coordinates: Experimental vs Fitted')
    # ax.grid(True)
    # ax.legend()

    # plt.tight_layout()
    # st.pyplot(fig)
    # plt.close(fig)

    # # Fitted formulas of the magnetic field components
    # # ----------------------------------------------------------------------------------
    # st.markdown("<h2 style='font-size:20px;'>3.1) General Form of Magnetic Field Components for this Model</h2>", unsafe_allow_html=True)
    # st.latex(r"B_r(r,\varphi) = A_{1r} r + B_{1r} r^2 \left[ C_{1r} + D_{1r} \sin(\alpha - \alpha_{0r}) + E_{1r} \cos(\alpha - \alpha_{0r}) \right]")
    # st.latex(r"B_y(r,\varphi) = A_y r^2 \left[ B_y + C_y \sin(\alpha - \alpha_{0y}) + D_y \cos(\alpha - \alpha_{0y}) \right] + E_y")
    # st.latex(r"B_\phi(r,\varphi) = A_{1\phi} r + B_{1\phi} r^2 \left[ C_{1\phi} + D_{1\phi} \sin(\alpha - \alpha_{0\phi}) + E_{1\phi} \cos(\alpha - \alpha_{0\phi}) \right]")

    # st.markdown("<h2 style='font-size:20px;'>3.2) Resulting Formulas</h2>", unsafe_allow_html=True)
    # st.latex(f"B_r = {sp.latex(Br_expr)}")
    # st.latex(f"B_y = {sp.latex(By_expr)}")
    # st.latex(f"B_\\phi = {sp.latex(Bphi_expr)}")
    # st.latex(r"\nabla \cdot \mathbf{B} = 0")

    # # Plot 7) Magnetic field representation in the cross section
    # # ----------------------------------------------------------------------------------
    # # --- Modelos ajustados del campo magnético ---
    # def Br_model_fitted(r, alpha):
    #     """
    #     Radial component with linear and quadratic radial terms and oscillatory part
    #     """
    #     linear_part = A1_Br * r
    #     quadratic_part = B1_Br * r**2
    #     oscillatory_part = C1_Br + D1_Br * np.sin(alpha - alpha0_Br) + E1_Br * np.cos(alpha - alpha0_Br)
    #     return linear_part + quadratic_part * oscillatory_part

    # def By_model_fitted(r, alpha):
    #     """
    #     Y component with quadratic radial term, oscillatory part, and offset
    #     """
    #     radial_part = A_By * r**2
    #     oscillatory_part = B_By + C_By * np.sin(alpha - alpha0_By) + D_By * np.cos(alpha - alpha0_By)
    #     return radial_part * oscillatory_part + E_By

    # def Bphi_model_fitted(r, alpha):
    #     """
    #     Azimuthal component with linear and quadratic radial terms and oscillatory part
    #     """
    #     linear_part = A1_Bphi * r
    #     quadratic_part = B1_Bphi * r**2
    #     oscillatory_part = C1_Bphi + D1_Bphi * np.sin(alpha - alpha0_Bphi) + E1_Bphi * np.cos(alpha - alpha0_Bphi)
    #     return linear_part + quadratic_part * oscillatory_part


    # # --- Crear la malla para la sección transversal ---
    # Npt = 300
    # x_min = min(x_ellipse_rotated) - 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
    # x_max = max(x_ellipse_rotated) + 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
    # y_min = min(y_ellipse_rotated) - 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
    # y_max = max(y_ellipse_rotated) + 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
    # X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, Npt), np.linspace(y_min, y_max, Npt))

    # # --- Parámetros de la elipse rotada ---
    # xc = np.mean(x_ellipse_rotated)
    # yc = np.mean(y_ellipse_rotated)

    # # Estimar semiejes y ángulo de rotación
    # ellipse_local_2d_rotated = np.column_stack((x_ellipse_rotated - xc, y_ellipse_rotated - yc))
    # ellipse_model = EllipseModel()
    # ellipse_model.estimate(ellipse_local_2d_rotated)
    # _, _, a_ellipse, b_ellipse, theta_ellipse = ellipse_model.params
    # delta = b_ellipse / a_ellipse
    # a_ell = 1  # Escala normalizada

    # # --- Calcular coordenadas elípticas ---
    # r_grid, alpha_grid = elliptical_coords(X_grid, Y_grid, xc, yc, a_ellipse, b_ellipse, theta_ellipse)

    # # --- Evaluar componentes del campo magnético ajustado en la malla ---
    # Br_vals_fitted = Br_model_fitted(r_grid, alpha_grid)
    # By_vals_fitted = By_model_fitted(r_grid, alpha_grid)
    # Bphi_vals_fitted = Bphi_model_fitted(r_grid, alpha_grid)

    # # --- Componentes métricos en la malla 2D ---
    # grr_corrected = a_ell**2 * (np.cos(alpha_grid)**2 + delta**2 * np.sin(alpha_grid)**2)
    # gyy_corrected = np.ones_like(r_grid)
    # gphiphi_corrected = a_ell**2 * r_grid**2 * (np.sin(alpha_grid)**2 + delta**2 * np.cos(alpha_grid)**2)
    # grphi_corrected = a_ell**2 * r_grid * np.sin(alpha_grid) * np.cos(alpha_grid) * (delta**2 - 1)

    # # --- Magnitud total del campo magnético ajustado en la malla ---
    # B_total_fitted = np.sqrt(
    #     grr_corrected * Br_vals_fitted**2 +
    #     gyy_corrected * By_vals_fitted**2 +
    #     gphiphi_corrected * Bphi_vals_fitted**2 +
    #     2 * grphi_corrected * Br_vals_fitted * Bphi_vals_fitted
    # )

    # # --- Enmascarar puntos fuera de la elipse (r > 1) ---
    # mask = (r_grid > 1.0)
    # Br_masked_fitted = np.ma.array(Br_vals_fitted, mask=mask)
    # By_masked_fitted = np.ma.array(By_vals_fitted, mask=mask)
    # Bphi_masked_fitted = np.ma.array(Bphi_vals_fitted, mask=mask)
    # Btotal_masked_fitted = np.ma.array(B_total_fitted, mask=mask)

    # # --- Gráficos 2D de contorno en Streamlit (2x2) ---
    # st.markdown("<h2 style='font-size:20px;'>3.3) Magnetic Field Cross Section (Fitted)</h2>", unsafe_allow_html=True)

    # fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2 filas, 2 columnas
    # fields_fitted = [Br_masked_fitted, By_masked_fitted, Bphi_masked_fitted, Btotal_masked_fitted]
    # titles = [r"$B_r$", r"$B_y$", r"$B_{\phi}$", r"$|\mathbf{B}|$"]

    # # Aplanar axs para iterar fácilmente
    # axs_flat = axs.flatten()

    # for ax, field, title in zip(axs_flat, fields_fitted, titles):
    #     ctf = ax.contourf(X_grid, Y_grid, field, 50, cmap='viridis')
    #     ax.contour(X_grid, Y_grid, field, 10, colors='k', alpha=0.4)
    #     ax.plot(x_ellipse_rotated, y_ellipse_rotated, 'k-', lw=2, label="Ellipse Boundary")
    #     ax.plot(x_traj_rotated, y_traj_rotated, 'b--', lw=2, label="Trajectory")
    #     ax.set_aspect('equal', 'box')
    #     ax.set_title(f"Fitted: {title}", fontsize=15)
    #     ax.set_xlabel("X local rotated")
    #     ax.set_ylabel("Y local rotated")
    #     fig.colorbar(ctf, ax=ax, shrink=0.9)
    #     ax.legend()

    # plt.tight_layout()
    # st.pyplot(fig)
    # plt.close(fig)


    # # Plot 8) Back to Local Components and GSE Components
    # # ----------------------------------------------------------------------------------
    # # --- Crear figura con dos subplots lado a lado en Streamlit ---
    # st.subheader("4) Fitted magnetic field in the Local and GSE Coordinate System")
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # # --- Gráfico en coordenadas cartesianas locales ---
    # ax1.plot(x_traj, B_vector, 'k-', linewidth=2.5, label=r'$|\mathbf{B}|$')  # Negro
    # ax1.plot(x_traj, Bx_traj, 'b-', linewidth=2.5, label=r'$B_x$')  # Azul fuerte
    # ax1.plot(x_traj, By_traj_cartesian, 'r-', linewidth=2.5, label=r'$B_y$')  # Rojo fuerte
    # ax1.plot(x_traj, Bz_traj, 'g-', linewidth=2.5, label=r'$B_z$')  # Verde fuerte
    # ax1.set_xlabel("X local rotated")
    # ax1.set_ylabel("Magnetic Field (Local Cartesian Components)")
    # ax1.set_title("Local Cartesian Coordinates")
    # ax1.legend(fontsize=14)  # Tamaño de fuente grande
    # ax1.grid(True)

    # # --- Gráfico en coordenadas GSE (si rotation_matrix está definida) ---
    # if Bx_GSE is not None:
    #     ax2.plot(x_traj, B_vector, 'k-', linewidth=2.5, label=r'$|\mathbf{B}|$')  # Negro
    #     ax2.plot(x_traj, Bx_GSE, 'b-', linewidth=2.5, label=r'$B_x$ GSE')  # Azul fuerte
    #     ax2.plot(x_traj, By_GSE, 'r-', linewidth=2.5, label=r'$B_y$ GSE')  # Rojo fuerte
    #     ax2.plot(x_traj, Bz_GSE, 'g-', linewidth=2.5, label=r'$B_z$ GSE')  # Verde fuerte
    #     ax2.set_xlabel("X local rotated")
    #     ax2.set_ylabel("Magnetic Field (GSE Components)")
    #     ax2.set_title("GSE Coordinates")
    #     ax2.legend(fontsize=14)  # Tamaño de fuente grande
    #     ax2.grid(True)
    # else:
    #     ax2.text(0.5, 0.5, "GSE coordinates unavailable\n(rotation_matrix not defined)", 
    #             ha='center', va='center', transform=ax2.transAxes)
    #     ax2.set_xlabel("X local rotated")
    #     ax2.set_title("GSE Coordinates")

    # plt.tight_layout()
    # st.pyplot(fig)
    # plt.close(fig)

    # # Plot 9) Fitting to the original GSE data
    # # ----------------------------------------------------------------------------------

    # B_vector = B_vector[::-1]
    # Bx_GSE = Bx_GSE[::-1]
    # By_GSE = By_GSE[::-1]
    # Bz_GSE = Bz_GSE[::-1]
    # adjusted_data = [B_vector, Bx_GSE, By_GSE, Bz_GSE]

    # # --- Gráfico comparativo en Streamlit ---
    # st.subheader("5) Fitting Results in GSE")
    # fig_compare, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # components = ['B', 'Bx', 'By', 'Bz']
    # data_compare = [B_data, Bx_data, By_data, Bz_data]  # Datos originales en GSE
    # titles_compare = [
    #     "Magnetic Field Intensity (B)",
    #     "Magnetic Field Component Bx",
    #     "Magnetic Field Component By",
    #     "Magnetic Field Component Bz",
    # ]

    # # Definir los puntos de inicio y fin del segmento
    # start_segment = ddoy_data[initial_point - 1]
    # end_segment = ddoy_data[final_point - 2]

    # for i, (component, orig_data, adj_data, title) in enumerate(
    #     zip(components, data_compare, adjusted_data, titles_compare)
    # ):
    #     # Plot datos experimentales como puntos negros
    #     ax[i].scatter(ddoy_data, orig_data, color='black', s=10, label=f'{component} Original')
        
    #     # Plot datos ajustados como línea roja discontinua (si están disponibles)
    #     if adj_data is not None:
    #         ax[i].plot(ddoy_data[initial_point - 1:final_point - 1], adj_data, 
    #                 color='red', linestyle='--', linewidth=2, label=f'{component} Fitted')
        
    #     # Líneas verticales para el inicio y fin del segmento
    #     ax[i].axvline(x=start_segment, color='gray', linestyle='--', label='Start of Segment')
    #     ax[i].axvline(x=end_segment, color='gray', linestyle='--', label='End of Segment')
        
    #     # Configuración del subplot
    #     ax[i].set_title(title, fontsize=18, fontweight='bold')
    #     ax[i].set_ylabel(f"{component} (nT)", fontsize=14)
    #     ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    #     ax[i].minorticks_on()
    #     ax[i].legend(fontsize=12)

    # ax[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
    # plt.tight_layout()
    # st.pyplot(fig_compare)
    # plt.close(fig_compare)

    # # --- Resultados de los parámetros de ajuste ---
    # st.markdown("<h2 style='font-size:20px;'>Fitting Parameters</h2>", unsafe_allow_html=True)

    # # Asumiendo que z0, angle_x, angle_y, angle_z, delta están definidos
    # # Si no, ajusta estos valores según tu contexto
    # st.latex(f"z_0 = {z0:.2f}")
    # st.latex(f"\\theta_x = {np.rad2deg(angle_x):.2f}^\\circ")
    # st.latex(f"\\theta_y = {np.rad2deg(angle_y):.2f}^\\circ")
    # st.latex(f"\\theta_z = {np.rad2deg(angle_z):.2f}^\\circ")
    # st.latex(f"\\delta = {delta:.2f}")


    # # Mostrar los resultados en Streamlit
    # st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results</h2>", unsafe_allow_html=True)
    # st.latex(f"R^2_B = {R2_B:.4f}")
    # st.latex(f"R^2_{{Bx}} = {R2_Bx:.4f}")
    # st.latex(f"R^2_{{By}} = {R2_By:.4f}")
    # st.latex(f"R^2_{{Bz}} = {R2_Bz:.4f}")
    # st.latex(f"R^2_{{avg}} = {R2_avg:.4f}")


    # # Plot 10) 3D interactive plot
    # # ----------------------------------------------------------------------------------
    # trajectory_3d(viz_3d_vars_opt)



    return (best_combination, B_components_fit, trajectory_vectors,
            viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt)









def trajectory_3d(viz_3d_vars_opt):
    # Unpack the variables from viz_3d_vars_opt
    (X_rot_opt, Y_rot_opt, Z_rot_opt,
     X_ellipse_opt, Z_ellipse_opt,
     X_intersect_opt, Z_intersect_opt,
     scale_factor_opt, a_opt, Z_max_opt,
     z_cut_opt, x1_opt, x2_opt,
     X_proj_ellipse, Y_proj_ellipse, Z_proj_ellipse,
     X_proj_inter, Y_proj_inter, Z_proj_inter,
     X_proj_traj, Y_proj_traj, Z_proj_traj,
     d, normalized_cylinder_axis) = viz_3d_vars_opt

    # 3D Figure Plot: 
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # st.subheader("Interactive 3D Visualization of the Flux Rope")
    # st.markdown(f"<h2 style='font-size:22px;'>Interactive 3D Visualization of the Flux Rope</h2>", unsafe_allow_html=True)


    fig_3d = go.Figure()

    fig_3d.add_trace(go.Surface(x=X_rot_opt, y=Y_rot_opt, z=Z_rot_opt,
                                colorscale='Blues', opacity=0.4, showscale=False, name="Flux Rope"))
    fig_3d.add_trace(go.Scatter3d(x=X_ellipse_opt, y=np.zeros_like(X_ellipse_opt), z=Z_ellipse_opt,
                                  mode='lines', line=dict(color='red', width=4), name="Scaled Ellipse"))
    fig_3d.add_trace(go.Scatter3d(x=X_intersect_opt, y=np.zeros_like(X_intersect_opt), z=Z_intersect_opt,
                                  mode='markers', marker=dict(size=8, color='red'), name="Scaled Intersection"))
    x_plane = np.linspace(-scale_factor_opt * a_opt * 1.5, scale_factor_opt * a_opt * 1.5, 10)
    z_plane = np.linspace(-Z_max_opt * 1.5, Z_max_opt * 1.5, 10)
    X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
    Y_plane = np.zeros_like(X_plane) 

    fig_3d.add_trace(go.Surface(x=X_plane, y=Y_plane, z=Z_plane,
                                colorscale='Greys', opacity=0.2, showscale=False, name="y=0 Plane"))
    fig_3d.add_trace(go.Scatter3d(x=[x1_opt, x2_opt], y=[0, 0], z=[z_cut_opt, z_cut_opt],
                                  mode='lines', line=dict(color='blue', width=4), name="Intersection Line"))
    x_extended = np.linspace(-scale_factor_opt * a_opt * 1.0, scale_factor_opt * a_opt * 1.0, 100)
    z_extended = np.full_like(x_extended, z_cut_opt)
    fig_3d.add_trace(go.Scatter3d(x=x_extended, y=np.zeros_like(x_extended), z=z_extended,
                                  mode='lines', line=dict(color='black', dash='dash', width=2), name="Extended Dashed Line"))
    fig_3d.add_trace(go.Scatter3d(x=X_proj_ellipse, y=Y_proj_ellipse, z=Z_proj_ellipse,
                                  mode='lines', line=dict(color='green', width=4), name="Projected Ellipse"))
    fig_3d.add_trace(go.Scatter3d(x=X_proj_inter, y=Y_proj_inter, z=Z_proj_inter,
                                  mode='markers', marker=dict(size=8, color='green'), name="Projected Intersection"))
    fig_3d.add_trace(go.Scatter3d(x=X_proj_traj, y=Y_proj_traj, z=Z_proj_traj,
                                  mode='lines', line=dict(color='magenta', width=4), name="Projected Trajectory"))
    x_range = np.linspace(-scale_factor_opt * a_opt, scale_factor_opt * a_opt, 10)
    y_range = np.linspace(-scale_factor_opt * a_opt, scale_factor_opt * a_opt, 10)
    X_trans, Y_trans = np.meshgrid(x_range, y_range)
    Z_trans = (d - normalized_cylinder_axis[0] * X_trans - normalized_cylinder_axis[1] * Y_trans) / normalized_cylinder_axis[2]
    fig_3d.add_trace(go.Surface(x=X_trans, y=Y_trans, z=Z_trans,
                                colorscale='YlOrBr', opacity=0.2, showscale=False, name="Transversal Plane"))

    # Definir límites del gráfico
    all_x = np.concatenate([X_rot_opt.flatten(), X_ellipse_opt, X_intersect_opt, X_plane.flatten(), [x1_opt, x2_opt], x_extended,
                            X_proj_ellipse, X_proj_inter, X_proj_traj, X_trans.flatten()])
    all_y = np.concatenate([Y_rot_opt.flatten(), np.zeros_like(X_ellipse_opt), np.zeros_like(X_intersect_opt), Y_plane.flatten(), [0, 0], np.zeros_like(x_extended),
                            Y_proj_ellipse, Y_proj_inter, Y_proj_traj, Y_trans.flatten()])
    all_z = np.concatenate([Z_rot_opt.flatten(), Z_ellipse_opt, Z_intersect_opt, Z_plane.flatten(), [z_cut_opt, z_cut_opt], z_extended,
                            Z_proj_ellipse, Z_proj_inter, Z_proj_traj, Z_trans.flatten()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    z_min, z_max = np.min(all_z), np.max(all_z)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    # Configurar el layout del gráfico 3D
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range], title="X"),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range], title="Y"),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range], title="Z"),
            aspectmode='cube'
        ),
        title="Cylinder with Cut, Projection, and Trajectory",
        showlegend=True
    )

    # Mostrar el gráfico 3D en Streamlit
    st.plotly_chart(fig_3d, use_container_width=True)




