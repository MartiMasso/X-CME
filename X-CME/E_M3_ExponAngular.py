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
    alpha = np.mod(alpha, 2*np.pi)
    return r, alpha

# Function to compute the fitting (with cache)
st.cache_data.clear()
@st.cache_data
def fit_M3_ExponAngular(data, initial_point, final_point, initial_date, final_date,  distance, lon_ecliptic):
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

    # Parámetros de prueba fijos (como en tu código original)
    z0_range = np.array([0.2])
    angle_x_range = np.array([np.radians(90)])
    angle_y_range = np.array([np.radians(0)])
    angle_z_range = np.array([-np.radians(0)])
    delta_range = np.array([0.7])

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
                        # Adjusted model functions to match the provided equations
                        def model_Br(r_alpha, A1_r, B1_r, C1_r, D1_r, E1_r, alpha0_r):
                            """
                            Radial component with linear and quadratic radial terms and oscillatory part
                            """
                            r, alpha = r_alpha
                            linear_part = A1_r * r
                            quadratic_part = B1_r * r**2
                            oscillatory_part = C1_r + D1_r * np.sin(alpha - alpha0_r) + E1_r * np.cos(alpha - alpha0_r)
                            return linear_part + quadratic_part * oscillatory_part

                        def model_By(r_alpha, A_y, B_y, C_y, D_y, alpha0_y, E_y):
                            """
                            Y component with quadratic radial term, oscillatory part, and offset
                            """
                            r, alpha = r_alpha
                            radial_part = A_y * r**2
                            oscillatory_part = B_y + C_y * np.sin(alpha - alpha0_y) + D_y * np.cos(alpha - alpha0_y)
                            return radial_part * oscillatory_part + E_y

                        def model_Bphi(r_alpha, A1_phi, B1_phi, C1_phi, D1_phi, E1_phi, alpha0_phi):
                            """
                            Azimuthal component with linear and quadratic radial terms and oscillatory part
                            """
                            r, alpha = r_alpha
                            linear_part = A1_phi * r
                            quadratic_part = B1_phi * r**2
                            oscillatory_part = C1_phi + D1_phi * np.sin(alpha - alpha0_phi) + E1_phi * np.cos(alpha - alpha0_phi)
                            return linear_part + quadratic_part * oscillatory_part

                        # Data preparation (keeping your original structure)
                        data_fit = np.vstack((r_vals, phi_vals)).T

                        # Initial guesses for the fitting
                        initial_guess_Br = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # [A1_r, B1_r, C1_r, D1_r, E1_r, alpha0_r]
                        initial_guess_By = [-1.0, 1.0, 0.1, 0.1, 0.0, 1.0]  # [A_y, B_y, C_y, D_y, alpha0_y, E_y]
                        initial_guess_Bphi = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # [A1_phi, B1_phi, C1_phi, D1_phi, E1_phi, alpha0_phi]

                        try:
                            # Fit for Br
                            params_Br, _ = curve_fit(model_Br, data_fit.T, Br_exp, p0=initial_guess_Br)
                            A1_Br, B1_Br, C1_Br, D1_Br, E1_Br, alpha0_Br = params_Br
                            Br_vector = model_Br(data_fit.T, A1_Br, B1_Br, C1_Br, D1_Br, E1_Br, alpha0_Br)

                            # Fit for By
                            params_By, _ = curve_fit(model_By, data_fit.T, By_exp_cyl, p0=initial_guess_By)  # Use By_exp_cyl
                            A_By, B_By, C_By, D_By, alpha0_By, E_By = params_By
                            By_vector = model_By(data_fit.T, A_By, B_By, C_By, D_By, alpha0_By, E_By)

                            # Fit for Bphi
                            params_Bphi, _ = curve_fit(model_Bphi, data_fit.T, Bphi_exp, p0=initial_guess_Bphi)
                            A1_Bphi, B1_Bphi, C1_Bphi, D1_Bphi, E1_Bphi, alpha0_Bphi = params_Bphi
                            Bphi_vector = model_Bphi(data_fit.T, A1_Bphi, B1_Bphi, C1_Bphi, D1_Bphi, E1_Bphi, alpha0_Bphi)

                            # R² calculations
                            ss_tot_Br = np.sum((Br_exp - np.mean(Br_exp))**2)
                            ss_res_Br = np.sum((Br_exp - Br_vector)**2)
                            R2_Br = 1 - (ss_res_Br / ss_tot_Br) if ss_tot_Br != 0 else 0

                            ss_tot_By = np.sum((By_exp_cyl - np.mean(By_exp_cyl))**2)
                            ss_res_By = np.sum((By_exp_cyl - By_vector)**2)
                            R2_By = 1 - (ss_res_By / ss_tot_By) if ss_tot_By != 0 else 0

                            ss_tot_Bphi = np.sum((Bphi_exp - np.mean(Bphi_exp))**2)
                            ss_res_Bphi = np.sum((Bphi_exp - Bphi_vector)**2)
                            R2_Bphi = 1 - (ss_res_Bphi / ss_tot_Bphi) if ss_tot_Bphi != 0 else 0

                            R2_avg = (R2_Br + R2_By + R2_Bphi) / 3

                            # Total magnetic field magnitude
                            B_vector = np.sqrt(
                                grr_traj * Br_vector**2 +
                                gyy_traj * By_vector**2 +
                                gphiphi_traj * Bphi_vector**2 +
                                2 * grphi_traj * Br_vector * Bphi_vector
                            )

                            # Print fitted parameters and R² values
                            print("\n--- Fitted Parameters ---")
                            print(f"Br: A1_r = {A1_Br:.6f}, B1_r = {B1_Br:.6f}, C1_r = {C1_Br:.6f}, D1_r = {D1_Br:.6f}, E1_r = {E1_Br:.6f}, alpha0_r = {alpha0_Br:.6f}, R² = {R2_Br:.6f}")
                            print(f"By: A_y = {A_By:.6f}, B_y = {B_By:.6f}, C_y = {C_By:.6f}, D_y = {D_By:.6f}, alpha0_y = {alpha0_By:.6f}, E_y = {E_By:.6f}, R² = {R2_By:.6f}")
                            print(f"Bphi: A1_phi = {A1_Bphi:.6f}, B1_phi = {B1_Bphi:.6f}, C1_phi = {C1_Bphi:.6f}, D1_phi = {D1_Bphi:.6f}, E1_phi = {E1_Bphi:.6f}, alpha0_phi = {alpha0_Bphi:.6f}, R² = {R2_Bphi:.6f}")
                            print(f"Average R² = {R2_avg:.6f}")

                        except RuntimeError as e:
                            st.write(f"Error in curve fitting: {str(e)}")

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

                            # Format parameters with 4 significant figures in scientific notation
                            A1_Br_rounded = f"{A1_Br:.4e}"
                            B1_Br_rounded = f"{B1_Br:.4e}"
                            C1_Br_rounded = f"{C1_Br:.4e}"
                            D1_Br_rounded = f"{D1_Br:.4e}"
                            alpha0_Br_rounded = f"{alpha0_Br:.4e}"
                            E1_Br_rounded = f"{E1_Br:.4e}"

                            A_By_rounded = f"{A_By:.4e}"
                            B_By_rounded = f"{B_By:.4e}"
                            C_By_rounded = f"{C_By:.4e}"
                            D_By_rounded = f"{D_By:.4e}"
                            alpha0_By_rounded = f"{alpha0_By:.4e}"
                            E_By_rounded = f"{E_By:.4e}"

                            A1_Bphi_rounded = f"{A1_Bphi:.4e}"
                            B1_Bphi_rounded = f"{B1_Bphi:.4e}"
                            C1_Bphi_rounded = f"{C1_Bphi:.4e}"
                            D1_Bphi_rounded = f"{D1_Bphi:.4e}"
                            alpha0_Bphi_rounded = f"{alpha0_Bphi:.4e}"
                            E1_Bphi_rounded = f"{E1_Bphi:.4e}"

                            params_Br_fit = (A1_Br, B1_Br, C1_Br, D1_Br, E1_Br, alpha0_Br)
                            params_By_fit = (A_By, B_By, C_By, D_By, alpha0_By, E_By)
                            params_Bphi_fit = (A1_Bphi, B1_Bphi, C1_Bphi, D1_Bphi, E1_Bphi, alpha0_Bphi)

                            # Define symbolic expressions using the formatted parameters
                            r, alpha = sp.symbols('r alpha')

                            # B_r expression
                            Br_expr = (sp.sympify(A1_Br_rounded) * r + sp.sympify(B1_Br_rounded) * r**2 * (
                                sp.sympify(C1_Br_rounded) +
                                sp.sympify(D1_Br_rounded) * sp.sin(alpha - sp.sympify(alpha0_Br_rounded)) +
                                sp.sympify(E1_Br_rounded) * sp.cos(alpha - sp.sympify(alpha0_Br_rounded))
                            ))

                            # B_y expression
                            By_expr = (sp.sympify(A_By_rounded) * r**2 * (
                                sp.sympify(B_By_rounded) +
                                sp.sympify(C_By_rounded) * sp.sin(alpha - sp.sympify(alpha0_By_rounded)) +
                                sp.sympify(D_By_rounded) * sp.cos(alpha - sp.sympify(alpha0_By_rounded))
                            )) + sp.sympify(E_By_rounded)

                            # B_phi expression
                            Bphi_expr = (sp.sympify(A1_Bphi_rounded) * r + sp.sympify(B1_Bphi_rounded) * r**2 * (
                                sp.sympify(C1_Bphi_rounded) +
                                sp.sympify(D1_Bphi_rounded) * sp.sin(alpha - sp.sympify(alpha0_Bphi_rounded)) +
                                sp.sympify(E1_Bphi_rounded) * sp.cos(alpha - sp.sympify(alpha0_Bphi_rounded))
                            ))


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
                            


    progress_bar.progress(100)
    progress_text.text("Processing complete! ✅")


    # POST PROCESSING
    ### (A). Data extraction
    if best_R2 is not None and best_combination is not None:
        st.success(f"Best R2 found: {best_R2:.4f}")

        # Plot 1: Oriented Flux Rope and intersection with plane y = 0
        X_rot_scaled, Y_rot_scaled, Z_rot_scaled, X_ellipse_scaled, Z_ellipse_scaled, X_intersections_scaled, Z_intersections_scaled, percentage_in_upper_half, z_cut_scaled, xs, ys, zs = plot1_vars

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



        # ----------------------------------------------------------------------------------
        # Plot 1) Oriented Flux Rope and intersection with plane y = 0
        # ----------------------------------------------------------------------------------
        st.subheader("1) Geometry of the fitted Flux Rope")

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
        ax1.plot([x_max_z, x_max_z], [0, 0], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
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
        ax2.plot([x_max_z, x_max_z], z_x_max_z_limits, 'b--', linewidth=1.5, label="maximum z")
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
        x_plane = np.linspace(- a * 1.5, a * 1.5, 10)
        z_plane = np.linspace(-Z_max_scaled * 1.5, Z_max_scaled * 1.5, 10)
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
        x_extended = np.linspace(-a * 1.0, a * 1.0, 100)
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
        x_range = np.linspace(-  a * 1.5,   a * 1.5, 10)
        y_range = np.linspace(-a_proj * 1.5, a_proj * 1.5, 10)
        X_trans, Y_trans = np.meshgrid(x_range, y_range)
        Z_trans = (d - axis_cylinder_norm[0] * X_trans - axis_cylinder_norm[1] * Y_trans) / axis_cylinder_norm[2]
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
        ax1.set_xlabel("Local X")
        ax1.set_ylabel("Local Y")
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
        ax2.set_xlabel("Rotated Local X")
        ax2.set_ylabel("Rotated Local Y")
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
        ax2.set_xlabel("Local X")
        ax2.set_ylabel("Local Y")
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

        # --- Plot GSE magnetic field components in local Cartesian coordinates ---
        ax1.plot(x_traj_GSE, B_GSE_exp_tot, 'k-', label=r'$|\mathbf{B}|$')
        ax1.plot(x_traj_GSE, Bx_GSE_exp, 'b-', label=r'$B_x$')
        ax1.plot(x_traj_GSE, By_GSE_exp, 'r-', label=r'$B_y$')
        ax1.plot(x_traj_GSE, Bz_GSE_exp, 'g-', label=r'$B_z$')
        ax1.set_xlabel("X local rotated")
        ax1.set_ylabel("In-situ GSE in x-Local Axis")
        ax1.set_title("GSE in x-Local")
        ax1.legend()
        ax1.grid(True)

        # --- Plot LOCAL magnetic field components in GSE coordinates ---
        ax2.plot(x_traj, Bx_Local_exp, 'b-', label=r"$B_x^L$")
        ax2.plot(x_traj, By_Local_exp, 'r-', label=r"$B_y^L$")
        ax2.plot(x_traj, Bz_Local_exp, 'g-', label=r"$B_z^L$")
        ax2.plot(x_traj, B_Local_total_exp, 'k--', label=r"$|\mathbf{B}|$")
        ax2.set_xlabel("X GSE axis")
        ax2.set_ylabel("In-situ GSE in x-GSE Axis")
        ax2.set_title("GSE Coordinates")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)
        
        # ----------------------------------------------------------------------------------
        # Plot 6) In-situ Cylindrical Components and fitted model
        # ----------------------------------------------------------------------------------

        st.subheader("6) In-Situ and Fitted Cylindrical Components")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # --- Plot (1): Cylindrical Coordinates for Exported Data ---
        ax1.plot(x_traj, B_total_exp_cyl, 'k-', label=r'$|\mathbf{B}|$')
        ax1.plot(x_traj, Br_exp, 'b-', label=r'$B^r$')
        ax1.plot(x_traj, By_exp_cyl, 'r-', label=r'$B^y$')
        ax1.plot(x_traj, Bphi_exp, 'g-', label=r'$B^\varphi$')
        ax1.set_xlabel('X local rotated')
        ax1.set_ylabel('Magnetic Field')
        ax1.set_title('Cylindrical Coordinates (Exported Data)')
        ax1.grid(True)
        ax1.legend()
        
        # --- Plot (2): Fitted Cylindrical Coordinates ---
        ax2.plot(x_traj, B_vector, 'k-', label=r'$|\mathbf{B}|$')
        ax2.plot(x_traj, Br_vector, 'b-', label=r'$B^r$')
        ax2.plot(x_traj, By_vector, 'r-', label=r'$B^y$')
        ax2.plot(x_traj, Bphi_vector, 'g-', label=r'$B^\varphi$')
        ax2.set_xlabel("X local rotated")
        ax2.set_ylabel("Magnetic Field (Cylindrical Components)")
        ax2.set_title("Fitted Cylindrical Coordinates")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # ----------------------------------------------------------------------------------
        # Plot 7) Magnetic field representation in the cross section
        # ----------------------------------------------------------------------------------

        # --- Modelos ajustados del campo magnético ---
        def Br_model_fitted(r, alpha):
            """
            Radial component with linear and quadratic radial terms and oscillatory part
            """
            linear_part = A1_Br * r
            quadratic_part = B1_Br * r**2
            oscillatory_part = C1_Br + D1_Br * np.sin(alpha - alpha0_Br) + E1_Br * np.cos(alpha - alpha0_Br)
            return linear_part + quadratic_part * oscillatory_part

        def By_model_fitted(r, alpha):
            """
            Y component with quadratic radial term, oscillatory part, and offset
            """
            radial_part = A_By * r**2
            oscillatory_part = B_By + C_By * np.sin(alpha - alpha0_By) + D_By * np.cos(alpha - alpha0_By)
            return radial_part * oscillatory_part + E_By

        def Bphi_model_fitted(r, alpha):
            """
            Azimuthal component with linear and quadratic radial terms and oscillatory part
            """
            linear_part = A1_Bphi * r
            quadratic_part = B1_Bphi * r**2
            oscillatory_part = C1_Bphi + D1_Bphi * np.sin(alpha - alpha0_Bphi) + E1_Bphi * np.cos(alpha - alpha0_Bphi)
            return linear_part + quadratic_part * oscillatory_part
        # ----------------------------------------------------------------------------------

        # --- Crear la malla para la sección transversal ---
        Npt = 300
        x_min = min(x_ellipse_rotated) - 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
        x_max = max(x_ellipse_rotated) + 0.1 * (max(x_ellipse_rotated) - min(x_ellipse_rotated))
        y_min = min(y_ellipse_rotated) - 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
        y_max = max(y_ellipse_rotated) + 0.1 * (max(y_ellipse_rotated) - min(y_ellipse_rotated))
        X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, Npt), np.linspace(y_min, y_max, Npt))

        # --- Parámetros de la elipse rotada ---
        xc = np.mean(x_ellipse_rotated)
        yc = np.mean(y_ellipse_rotated)

        # Estimar semiejes y ángulo de rotación
        ellipse_local_2d_rotated = np.column_stack((x_ellipse_rotated - xc, y_ellipse_rotated - yc))
        ellipse_model = EllipseModel()
        ellipse_model.estimate(ellipse_local_2d_rotated)
        _, _, a_ellipse, b_ellipse, theta_ellipse = ellipse_model.params
        delta = b_ellipse / a_ellipse
        a_ell = 1  # Escala normalizada

        # --- Calcular coordenadas elípticas ---
        r_grid, alpha_grid = elliptical_coords(X_grid, Y_grid, xc, yc, a_ellipse, b_ellipse, theta_ellipse)

        # --- Evaluar componentes del campo magnético ajustado en la malla ---
        Br_vals_fitted = Br_model_fitted(r_grid, alpha_grid)
        By_vals_fitted = By_model_fitted(r_grid, alpha_grid)
        Bphi_vals_fitted = Bphi_model_fitted(r_grid, alpha_grid)

        # --- Componentes métricos en la malla 2D ---
        grr_corrected = a_ell**2 * (np.cos(alpha_grid)**2 + delta**2 * np.sin(alpha_grid)**2)
        gyy_corrected = np.ones_like(r_grid)
        gphiphi_corrected = a_ell**2 * r_grid**2 * (np.sin(alpha_grid)**2 + delta**2 * np.cos(alpha_grid)**2)
        grphi_corrected = a_ell**2 * r_grid * np.sin(alpha_grid) * np.cos(alpha_grid) * (delta**2 - 1)

        # --- Magnitud total del campo magnético ajustado en la malla ---
        B_total_fitted = np.sqrt(
            grr_corrected * Br_vals_fitted**2 +
            gyy_corrected * By_vals_fitted**2 +
            gphiphi_corrected * Bphi_vals_fitted**2 +
            2 * grphi_corrected * Br_vals_fitted * Bphi_vals_fitted
        )

        # --- Enmascarar puntos fuera de la elipse (r > 1) ---
        mask = (r_grid > 1.0)
        Br_masked_fitted = np.ma.array(Br_vals_fitted, mask=mask)
        By_masked_fitted = np.ma.array(By_vals_fitted, mask=mask)
        Bphi_masked_fitted = np.ma.array(Bphi_vals_fitted, mask=mask)
        Btotal_masked_fitted = np.ma.array(B_total_fitted, mask=mask)

        # --- Gráficos 2D de contorno en Streamlit (2x2) ---
        st.subheader("7) Magnetic Field Cross Section (Fitted)")

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2 filas, 2 columnas
        fields_fitted = [Br_masked_fitted, By_masked_fitted, Bphi_masked_fitted, Btotal_masked_fitted]
        titles = [r"$B_r$", r"$B_y$", r"$B_{\phi}$", r"$|\mathbf{B}|$"]

        # Aplanar axs para iterar fácilmente
        axs_flat = axs.flatten()

        for ax, field, title in zip(axs_flat, fields_fitted, titles):
            ctf = ax.contourf(X_grid, Y_grid, field, 50, cmap='viridis')
            ax.contour(X_grid, Y_grid, field, 10, colors='k', alpha=0.4)
            ax.plot(x_ellipse_rotated, y_ellipse_rotated, 'k-', lw=2, label="Ellipse Boundary")
            ax.plot(x_traj_rotated, y_traj_rotated, 'b--', lw=2, label="Trajectory")
            ax.set_aspect('equal', 'box')
            ax.set_title(f"Fitted: {title}", fontsize=15)
            ax.set_xlabel("X local rotated")
            ax.set_ylabel("Y local rotated")
            fig.colorbar(ctf, ax=ax, shrink=0.9)
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        # Fitted formulas of the magnetic field components
        # ----------------------------------------------------------------------------------
        st.markdown("<h2 style='font-size:20px;'>3.1) General Form of Magnetic Field Components for this Model</h2>", unsafe_allow_html=True)
        st.latex(r"B_r(r,\varphi) = A_{1r} r + B_{1r} r^2 \left[ C_{1r} + D_{1r} \sin(\alpha - \alpha_{0r}) + E_{1r} \cos(\alpha - \alpha_{0r}) \right]")
        st.latex(r"B_y(r,\varphi) = A_y r^2 \left[ B_y + C_y \sin(\alpha - \alpha_{0y}) + D_y \cos(\alpha - \alpha_{0y}) \right] + E_y")
        st.latex(r"B_\phi(r,\varphi) = A_{1\phi} r + B_{1\phi} r^2 \left[ C_{1\phi} + D_{1\phi} \sin(\alpha - \alpha_{0\phi}) + E_{1\phi} \cos(\alpha - \alpha_{0\phi}) \right]")

        st.markdown("<h2 style='font-size:20px;'>3.2) Resulting Formulas</h2>", unsafe_allow_html=True)
        st.latex(f"B_r = {sp.latex(Br_expr)}")
        st.latex(f"B_y = {sp.latex(By_expr)}")
        st.latex(f"B_\\phi = {sp.latex(Bphi_expr)}")
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

        B_vector = B_vector[::-1]
        Bx_GSE = Bx_GSE[::-1]
        By_GSE = By_GSE[::-1]
        Bz_GSE = Bz_GSE[::-1]
        adjusted_data = [B_vector, Bx_GSE, By_GSE, Bz_GSE]

        # --- Gráfico comparativo en Streamlit ---
        st.subheader("9) Fitting to the original Data")
        fig_compare, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        components = ['B', 'Bx', 'By', 'Bz']
        data_compare = [B_data, Bx_data, By_data, Bz_data]  # Datos originales en GSE
        titles_compare = [
            "Magnetic Field Intensity (B)",
            "Magnetic Field Component Bx",
            "Magnetic Field Component By",
            "Magnetic Field Component Bz",
        ]

        # Definir los puntos de inicio y fin del segmento
        start_segment = ddoy_data[initial_point - 1]
        end_segment = ddoy_data[final_point - 2]

        for i, (component, orig_data, adj_data, title) in enumerate(
            zip(components, data_compare, adjusted_data, titles_compare)
        ):
            # Plot datos experimentales como puntos negros
            ax[i].scatter(ddoy_data, orig_data, color='black', s=10, label=f'{component} Original')
            
            # Plot datos ajustados como línea roja discontinua (si están disponibles)
            if adj_data is not None:
                ax[i].plot(ddoy_data[initial_point - 1:final_point - 1], adj_data, 
                        color='red', linestyle='--', linewidth=2, label=f'{component} Fitted')
            
            # Líneas verticales para el inicio y fin del segmento
            ax[i].axvline(x=start_segment, color='gray', linestyle='--', label='Start of Segment')
            ax[i].axvline(x=end_segment, color='gray', linestyle='--', label='End of Segment')
            
            # Configuración del subplot
            ax[i].set_title(title, fontsize=18, fontweight='bold')
            ax[i].set_ylabel(f"{component} (nT)", fontsize=14)
            ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            ax[i].minorticks_on()
            ax[i].legend(fontsize=12)

        ax[-1].set_xlabel("Day of the Year (ddoy)", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_compare)
        plt.close(fig_compare)

        # --- Resultados de los parámetros de ajuste ---
        st.markdown("<h2 style='font-size:20px;'>Fitting Parameters</h2>", unsafe_allow_html=True)

        # Asumiendo que z0, angle_x, angle_y, angle_z, delta están definidos
        # Si no, ajusta estos valores según tu contexto
        st.latex(f"z_0 = {z0:.2f}")
        st.latex(f"\\theta_x = {np.rad2deg(angle_x):.2f}^\\circ")
        st.latex(f"\\theta_y = {np.rad2deg(angle_y):.2f}^\\circ")
        st.latex(f"\\theta_z = {np.rad2deg(angle_z):.2f}^\\circ")
        st.latex(f"\\delta = {delta:.2f}")

        # Mostrar los resultados en Streamlit
        st.markdown("<h2 style='font-size:20px;'>Goodness-of-Fit (R²) Results</h2>", unsafe_allow_html=True)
        st.latex(f"R^2_B = {R2_B:.4f}")
        st.latex(f"R^2_{{Bx}} = {R2_Bx:.4f}")
        st.latex(f"R^2_{{By}} = {R2_By:.4f}")
        st.latex(f"R^2_{{Bz}} = {R2_Bz:.4f}")
        st.latex(f"R^2_{{avg}} = {R2_avg:.4f}")

        # ----------------------------------------------------------------------------------
        # Plot 10) Current Density formulas and Plot in the Cross section
        # ----------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------
        # Plot 10) Current Density formulas and Plot in the Cross section
        # # ----------------------------------------------------------------------------------

        # st.subheader("10) Current Density in the Cross Section")

        # # --- Constants ---
        # mu_0 = 4e-7 * np.pi  # Vacuum permeability in H/m (4π × 10⁻⁷)

        # # --- Fitted parameters from Model 2 (rounded to 3 significant figures) ---
        # # These are already computed in your script; we’ll use them directly
        # A1_r = float(f"{A1_Br:.3g}")
        # B1_r = float(f"{B1_Br:.3g}")
        # C1_r = float(f"{C1_Br:.3g}")
        # D1_r = float(f"{D1_Br:.3g}")
        # E1_r = float(f"{E1_Br:.3g}")
        # alpha0_r = float(f"{alpha0_Br:.3g}")

        # A_y = float(f"{A_By:.3g}")
        # B_y = float(f"{B_By:.3g}")
        # C_y = float(f"{C_By:.3g}")
        # D_y = float(f"{D_By:.3g}")
        # alpha0_y = float(f"{alpha0_By:.3g}")
        # E_y = float(f"{E_By:.3g}")

        # A1_phi = float(f"{A1_Bphi:.3g}")
        # B1_phi = float(f"{B1_Bphi:.3g}")
        # C1_phi = float(f"{C1_Bphi:.3g}")
        # D1_phi = float(f"{D1_Bphi:.3g}")
        # E1_phi = float(f"{E1_Bphi:.3g}")
        # alpha0_phi = float(f"{alpha0_Bphi:.3g}")

        # # --- Current density formulas with substituted parameters ---
        # # j_r = (A_y * r / mu_0) * [C_y * cos(phi - alpha0_y) - D_y * sin(phi - alpha0_y)]
        # jr_expr = f"({A_y:.3g} * r / {mu_0:.3g}) * ({C_y:.3g} * cos(phi - {alpha0_y:.3g}) - {D_y:.3g} * sin(phi - {alpha0_y:.3g}))"

        # # j_phi = -(2 * A_y * r / mu_0) * [B_y + C_y * sin(phi - alpha0_y) + D_y * cos(phi - alpha0_y)]
        # jphi_expr = f"(-2 * {A_y:.3g} * r / {mu_0:.3g}) * ({B_y:.3g} + {C_y:.3g} * sin(phi - {alpha0_y:.3g}) + {D_y:.3g} * cos(phi - {alpha0_y:.3g}))"

        # # j_z = (1 / mu_0) * {2 * A1_phi + 3 * B1_phi * r * [C1_phi + D1_phi * sin(phi - alpha0_phi) + E1_phi * cos(phi - alpha0_phi)] 
        # #                     - B1_r * r * [D1_r * cos(phi - alpha0_r) - E1_r * sin(phi - alpha0_r)]}
        # jz_expr = (f"(1 / {mu_0:.3g}) * (2 * {A1_phi:.3g} + 3 * {B1_phi:.3g} * r * ({C1_phi:.3g} + {D1_phi:.3g} * sin(phi - {alpha0_phi:.3g}) + {E1_phi:.3g} * cos(phi - {alpha0_phi:.3g})) "
        #         f"- {B1_r:.3g} * r * ({D1_r:.3g} * cos(phi - {alpha0_r:.3g}) - {E1_r:.3g} * sin(phi - {alpha0_r:.3g})))")

        # # --- Display current density formulas in LaTeX ---
        # st.markdown("<h2 style='font-size:20px;'>10.1) Current Density Formulas</h2>", unsafe_allow_html=True)
        # st.latex(r"j_r = \frac{A_y r}{\mu_0} \left[ C_y \cos(\phi - \alpha_{0y}) - D_y \sin(\phi - \alpha_{0y}) \right]")
        # st.latex(f"j_r = {jr_expr}")
        # st.latex(r"j_\phi = -\frac{2 A_y r}{\mu_0} \left[ B_y + C_y \sin(\phi - \alpha_{0y}) + D_y \cos(\phi - \alpha_{0y}) \right]")
        # st.latex(f"j_\phi = {jphi_expr}")
        # st.latex(r"j_z = \frac{1}{\mu_0} \left( 2 A_{1\phi} + 3 B_{1\phi} r \left[ C_{1\phi} + D_{1\phi} \sin(\phi - \alpha_{0\phi}) + E_{1\phi} \cos(\phi - \alpha_{0\phi}) \right] - B_{1r} r \left[ D_{1r} \cos(\phi - \alpha_{0r}) - E_{1r} \sin(\phi - \alpha_{0r}) \right] \right)")
        # st.latex(f"j_z = {jz_expr}")

        # # --- Define numerical functions for current density ---
        # def jr_func(r, phi):
        #     return (A_y * r / mu_0) * (C_y * np.cos(phi - alpha0_y) - D_y * np.sin(phi - alpha0_y))

        # def jphi_func(r, phi):
        #     return (-2 * A_y * r / mu_0) * (B_y + C_y * np.sin(phi - alpha0_y) + D_y * np.cos(phi - alpha0_y))

        # def jz_func(r, phi):
        #     term1 = 2 * A1_phi
        #     term2 = 3 * B1_phi * r * (C1_phi + D1_phi * np.sin(phi - alpha0_phi) + E1_phi * np.cos(phi - alpha0_phi))
        #     term3 = -B1_r * r * (D1_r * np.cos(phi - alpha0_r) - E1_r * np.sin(phi - alpha0_r))
        #     return (1 / mu_0) * (term1 + term2 + term3)

        # # --- Evaluate current density on the grid (reusing grid from Plot 7) ---
        # jr_vals = jr_func(r_grid, alpha_grid)
        # jphi_vals = jphi_func(r_grid, alpha_grid)
        # jz_vals = jz_func(r_grid, alpha_grid)

        # # --- Compute total current density magnitude ---
        # j_total = np.sqrt(jr_vals**2 + jphi_vals**2 + jz_vals**2)

        # # --- Mask points outside the ellipse (r > 1) ---
        # jr_masked = np.ma.array(jr_vals, mask=mask)
        # jphi_masked = np.ma.array(jphi_vals, mask=mask)
        # jz_masked = np.ma.array(jz_vals, mask=mask)
        # jtotal_masked = np.ma.array(j_total, mask=mask)

        # # --- Plot current density components ---
        # st.markdown("<h2 style='font-size:20px;'>10.2) Current Density Cross Section</h2>", unsafe_allow_html=True)
        # fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 grid
        # fields_j = [jr_masked, jphi_masked, jz_masked, jtotal_masked]
        # titles_j = [r"$j_r$", r"$j_\phi$", r"$j_z$", r"$|\mathbf{j}|$"]

        # axs_flat = axs.flatten()
        # for ax, field, title in zip(axs_flat, fields_j, titles_j):
        #     ctf = ax.contourf(X_grid, Y_grid, field, 50, cmap='plasma')
        #     ax.contour(X_grid, Y_grid, field, 10, colors='k', alpha=0.4)
        #     ax.plot(x_ellipse_rotated, y_ellipse_rotated, 'k-', lw=2, label="Ellipse Boundary")
        #     ax.plot(x_traj_rotated, y_traj_rotated, 'b--', lw=2, label="Trajectory")
        #     ax.set_aspect('equal', 'box')
        #     ax.set_title(f"Current Density: {title}", fontsize=15)
        #     ax.set_xlabel("X local rotated")
        #     ax.set_ylabel("Y local rotated")
        #     fig.colorbar(ctf, ax=ax, shrink=0.9, label="A/m²")
        #     ax.legend()

        # plt.tight_layout()
        # st.pyplot(fig)
        # plt.close(fig)



    if best_combination is None:
        return None

    return (best_combination, B_components_fit, trajectory_vectors,
            viz_3d_vars_opt, viz_2d_local_vars_opt, viz_2d_rotated_vars_opt)
