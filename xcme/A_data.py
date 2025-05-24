import pandas as pd
import numpy as np
import streamlit as st
import os
import heliosat
from heliosat.util import dt_utc_from_str
import matplotlib.pyplot as plt
import math
import astropy.units as u
from sunpy.coordinates import HeliocentricEarthEcliptic, get_horizons_coord, get_body_heliographic_stonyhurst
from sunpy.time import parse_time

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

def handle_file_upload(upload_option):
    """
    Manages the upload of local files or the download of data from the cloud within a defined interval.
    Allows selection of start and end date/time and choice between STA, Wind, SOLO, PSP satellites.
    For uploaded files, automatically detects the delimiter (comma or whitespace) and retains only the columns:
    ddoy, B, Bx, By, Bz, Np, Vsw, Vth (or, for RTN, ddoy, B, Br, Bt, Bn, Np, Vsw, Vth).
    For cloud downloads, generates a DataFrame with data from the selected satellite.
    """
    data, file_name = None, None

    if upload_option == "Upload a file":
        # Option to upload a local file
        uploaded_file = st.session_state.get("uploaded_example") or st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt", "ascii"])
        if uploaded_file:
            file_name = uploaded_file.name
            uploaded_file.seek(0)
            sample = uploaded_file.read(1024)
            try:
                sample_str = sample.decode("utf-8")
            except AttributeError:
                sample_str = sample
            uploaded_file.seek(0)
            
            if ',' in sample_str:
                data = pd.read_csv(uploaded_file, sep=',')
            else:
                ext = file_name.split('.')[-1].lower()
                if ext == 'csv':
                    data = pd.read_csv(uploaded_file, delim_whitespace=True)
                elif ext in ['txt', 'ascii']:
                    data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
                    if data.shape[1] == 7:
                        data.columns = ['ddoy', 'B', 'Bx', 'By', 'Bz', 'Np', 'Vth']
                    elif data.shape[1] == 8:
                        data.columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn', 'Np', 'Vth', 'Vsw']
                    else:
                        st.error("Unexpected file format. Check the number of columns.")
                        return None, None
                else:
                    st.error("Unsupported file format. Please upload a CSV or TXT file.")
                    return None, None

            if 'time' in data.columns:
                data['time'] = pd.to_datetime(data['time'], utc=True)
            
            if {'Bx', 'By', 'Bz'}.issubset(data.columns):
                desired_columns = ['ddoy', 'B', 'Bx', 'By', 'Bz', 'Np', 'Vsw', 'Vth']
            elif {'Br', 'Bt', 'Bn'}.issubset(data.columns):
                desired_columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn', 'Np', 'Vsw', 'Vth']
            else:
                desired_columns = ['ddoy', 'B', 'Np', 'Vsw', 'Vth']
            
            data = data[[col for col in desired_columns if col in data.columns]]

    elif upload_option == "Download from cloud":
        st.write("Select the satellite and time interval to download data from the cloud:")

        # Select satellite
        satellite = st.selectbox("Select satellite", ["Wind", "STA", "SOLO", "PSP"])

        # Select start and end date/time
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
            start_time = st.time_input("Start time", value=pd.to_datetime("00:00").time())
        with col2:
            end_date = st.date_input("End date", value=pd.to_datetime("2020-01-02"))
            end_time = st.time_input("End time", value=pd.to_datetime("00:00").time())

        if start_date and start_time and end_date and end_time:
            # Combine date and time into a single datetime object (UTC)
            start_datetime = pd.to_datetime(f"{start_date} {start_time}", utc=True)
            end_datetime = pd.to_datetime(f"{end_date} {end_time}", utc=True)

            # Validate that start date/time is earlier than end date/time
            if start_datetime >= end_datetime:
                st.error("The start date and time must be earlier than the end date and time.")
                return None, None

            # Convert dates for download with heliosat
            t_s = dt_utc_from_str(start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
            t_e = dt_utc_from_str(end_datetime.strftime("%Y-%m-%d %H:%M:%S"))

            # Map selected satellite to its heliosat class
            satellite_map = {
                "Wind": heliosat.Wind,
                "STA": heliosat.STA,
                "SOLO": heliosat.SOLO,
                "PSP": heliosat.PSP
            }

            # Map parameters for magnetic and plasma data
            mag_param_map = {
                "Wind": "wind_mfi_h2",       # High-resolution magnetic field
                "STA": "sta_impact_l1_mag",  # Level 1 magnetic field
                "SOLO": "solo_mag",          # Magnetic field
                "PSP": "mag"                 # Assumed, to be confirmed
            }
            plasma_param_map = {
                "Wind": "wind_swe_h1",        # Solar wind plasma
                "STA": "sta_plastic_l2_proton",  # Proton data
                "SOLO": None,                 # No plasma key available in output
                "PSP": "sweap"                # Assumed, to be confirmed
            }

            # Initialize the satellite with error handling for PSP
            try:
                sat = satellite_map[satellite]()
            except Exception as e:
                st.error(f"Failed to initialize {satellite}: {str(e)}. Check kernel availability.")
                return None, None

            # Download magnetic field data
            try:
                t_mag, b_mag = sat.get(
                    [t_s, t_e],
                    mag_param_map[satellite],
                    as_endpoints=True,
                    frame="HEEQ",
                    return_datetimes=True
                )
            except KeyError as e:
                st.error(f"Error: Magnetic field data ('{mag_param_map[satellite]}') not available for {satellite}: {str(e)}")
                return None, None
            except Exception as e:
                st.error(f"Error retrieving magnetic field data for {satellite}: {str(e)}")
                return None, None

            # Download solar wind data (optional)
            t_swe, m_swe = None, None
            plasma_param = plasma_param_map[satellite]
            if plasma_param:
                try:
                    t_swe, m_swe = sat.get(
                        [t_s, t_e],
                        plasma_param,
                        as_endpoints=True,
                        return_datetimes=True
                    )
                except KeyError as e:
                    st.warning(f"Plasma data ('{plasma_param}') not available for {satellite}: {str(e)}. Using magnetic field data only.")
                except Exception as e:
                    st.warning(f"Error retrieving plasma data for {satellite}: {str(e)}. Using magnetic field data only.")

            # Convert times to datetime (UTC)
            t_mag = pd.to_datetime(t_mag, utc=True)

            # Convert arrays to numpy
            b_mag = np.array(b_mag)  # (N, 3): Bx, By, Bz

            # Create DataFrame for magnetic field data
            mag_data = pd.DataFrame({
                "time": t_mag,
                "Bx": b_mag[:, 0],
                "By": b_mag[:, 1],
                "Bz": b_mag[:, 2]
            })

            # Process solar wind data if available
            if t_swe is not None and m_swe is not None:
                t_swe = pd.to_datetime(t_swe, utc=True)
                m_swe = np.array(m_swe)  # (N, 3): Density, Velocity, Temperature
                swe_data = pd.DataFrame({
                    "time": t_swe,
                    "Np": m_swe[:, 0],   # Proton density
                    "Vsw": m_swe[:, 1],  # Solar wind velocity
                    "Vth": m_swe[:, 2]   # Thermal velocity
                })

                # Merge both DataFrames
                data = pd.merge_asof(
                    mag_data.sort_values("time"),
                    swe_data.sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta("1 minute")
                )
            else:
                data = mag_data  # Use only magnetic data if plasma is unavailable

            # Resample to get 1 data point every 15 minutes
            data = data.set_index("time")
            data = data.resample("15T").mean()

            # Calculate the ddoy column (day of year with fraction)
            data["ddoy"] = (
                data.index.dayofyear
                + data.index.hour / 24
                + data.index.minute / (24 * 60)
                + data.index.second / (24 * 3600)
            )

            # Calculate the magnetic field magnitude (B)
            data["B"] = np.sqrt(data["Bx"]**2 + data["By"]**2 + data["Bz"]**2)

            # Retain only the columns of interest (adjust based on availability)
            available_columns = ["ddoy", "B", "Bx", "By", "Bz"]
            if "Np" in data.columns:
                available_columns.extend(["Np", "Vsw", "Vth"])
            data = data[available_columns]

            # Reset the index
            data = data.reset_index(drop=True)

            # Prefix map for file naming
            prefix_map = {
                "Wind": "WI",
                "STA": "STA",
                "SOLO": "SO",
                "PSP": "PSP"
            }
            
            # Calculate day of year (DOY) from start_datetime
            doy = start_datetime.timetuple().tm_yday  # Day of year as integer (1-366)
            doy_str = f"{doy:03d}"  # Format as three-digit string (e.g., "273")
            
            # Generate the file name with prefix, date, DOY, and sampling interval
            prefix = prefix_map.get(satellite, satellite)  # Fallback to satellite name if not in map
            file_name = f"{prefix}_{start_datetime.strftime('%Y%m%d')}_{doy_str}_30m.csv"

    # Check for NaN values
    if data is not None:
        if data.isna().any().any():
            st.warning("The file contains NaN values.")
        else:
            st.success("The file contains no NaN values.")

    return data, file_name

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

def resample_data(data):
    initial_points = len(data)
    # Si hay más de 1000 puntos, el step es el techo de (puntos / 1000); si no, se conserva 1
    step = math.ceil(initial_points / 350) if initial_points > 350 else 1
    if step > 1:
        st.write(f"Applying resampling: we take 1 for each {step} points.")
        data = data.iloc[::step].reset_index(drop=True)
        st.write(f"Number of points after resampling: {len(data)}")
    return data


def load_data(file):
    """
    Loads a CSV or TXT file and returns a pandas DataFrame.
    Supports both space-separated and comma-separated data.
    Reorders columns to: 
      - GSE: ddoy, B, Bx, By, Bz, Np, Vth, Vsw 
      - RTN: ddoy, B, Br, Bt, Bn, Np, Vth, Vsw
    It also adds missing optional columns with NaN, checks for inf/NaN, 
    interpolates NaN values, and performs resampling if the number of points exceeds certain thresholds.
    """
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        # Attempt to read with spaces first
        try:
            data = pd.read_csv(file, delim_whitespace=True, header=None)
            file.seek(0)
            if data.iloc[0].astype(str).str.match(r'[a-zA-Z]').any():
                data = pd.read_csv(file, delim_whitespace=True)
        except ValueError:
            file.seek(0)
            try:
                data = pd.read_csv(file, sep=',')
            except Exception as e:
                raise ValueError(f"Error loading data with comma separator: {e}")
        
        # Assign column names if no header
        if isinstance(data.columns[0], int):
            if data.shape[1] == 5:
                data.columns = ['ddoy', 'B', 'Bx', 'By', 'Bz']
            elif data.shape[1] == 7:
                data.columns = ['ddoy', 'B', 'Bx', 'By', 'Bz', 'Np', 'Vth']
            elif data.shape[1] == 8:
                data.columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn', 'Np', 'Vth', 'Vsw']
            elif data.shape[1] >= 14:
                data.columns = ['Br', 'Bt', 'Bn', 'B', 'ddoy', 'Np', 'Vth', 'Vr', 'Vt', 'Vn', 'QF', 'Vsw', 'Beta_p', 'date']
            else:
                raise ValueError("Unexpected file format. Check the number of columns.")
        else:
            expected_cols = {'ddoy', 'B', 'Bx', 'By', 'Bz', 'Br', 'Bt', 'Bn', 'Np', 'Vth', 'Vsw', 'date'}
            if not any(col in expected_cols for col in data.columns):
                raise ValueError("File columns do not match expected formats.")
        
        # Count initial points
        initial_points = len(data)
        st.write(f"### Number of initial points: {initial_points}")
        
        # Correction of NaN and inf values
        # Replace inf with NaN and report
        inf_check = data.replace([np.inf, -np.inf], np.nan).isna().sum()
        if inf_check.any():
            st.warning("The following columns contain inf or NaN before interpolation:")
            st.write(inf_check[inf_check > 0])
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.interpolate(method='linear', limit_direction='both')
            
            # Check after interpolation
            remaining_nans = data.isna().sum()
            if remaining_nans.any():
                st.warning("There are still NaN values after interpolation in the following columns:")
                st.write(remaining_nans[remaining_nans > 0])
                data = data.fillna(0)  # Fill remaining NaNs with 0
                st.success("All remaining NaN values have been filled with 0.")
            else:
                st.success("All inf/NaN values have been successfully interpolated.")
        else:
            st.success("No inf or NaN values were detected in the file.")
        
        if data.isna().any().any():
            raise ValueError("Data contains NaN values. Please check the preprocessing steps.")

        # Add missing optional columns with NaN
        for col in ['Np', 'Vth', 'Vsw']:
            if col not in data.columns:
                data[col] = np.nan

        # Identify coordinate system
        coordinate_system = identify_coordinate_system(data)
        st.write(f"### Coordinate System: **{coordinate_system}**")
        
        # Reorganize columns
        if coordinate_system == "GSE":
            desired_columns = ['ddoy', 'B', 'Bx', 'By', 'Bz', 'Np', 'Vth', 'Vsw']
        elif coordinate_system == "RTN":
            desired_columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn', 'Np', 'Vth', 'Vsw']
        else:
            raise ValueError("Could not determine coordinate system.")
        
        available_columns = [col for col in desired_columns if col in data.columns]
        data = data[available_columns]
        
        display_file_details(file.name, coordinate_system)
        display_data_info(data, coordinate_system)
        st.write(data.shape)    # Aquí deberías ver la nueva forma

        
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def identify_coordinate_system(data):
    """
    Identifica si el sistema de coordenadas es RTN o GSE según las columnas.
    """
    if {'Br', 'Bt', 'Bn'}.issubset(data.columns):
        return "RTN"
    elif {'Bx', 'By', 'Bz'}.issubset(data.columns):
        return "GSE"
    else:
        return "Unknown"

def doy_2_datetime(ddoy, year):
    """
    Convierte un día del año (DOY) a formato de fecha y hora.
    """
    doy = int(ddoy)
    day_fraction = ddoy - doy
    hr = int(24 * day_fraction)
    min = int((24 * day_fraction - hr) * 60)
    sec = int(((24 * day_fraction - hr) * 60 - min) * 60)
    dt = pd.to_datetime(f"{year}{str(doy).zfill(3)}{str(hr).zfill(2)}{str(min).zfill(2)}{str(sec).zfill(2)}",
                         format="%Y%j%H%M%S")
    return dt


def get_position_data(body_name, coord, obstime):
    """
    Transforms the given coordinates to the HeliocentricEarthEcliptic frame and returns 
    the radial distance (in AU) and the ecliptic longitude (in degrees).
    """
    # Set the HeliocentricEarthEcliptic frame with the event time
    hee_frame = HeliocentricEarthEcliptic(obstime=obstime)
    coord_hee = coord.transform_to(hee_frame)
    # Get the distance and longitude
    distance = coord_hee.spherical.distance.to(u.AU).value
    lon = coord_hee.spherical.lon.to(u.deg).value % 360  # Ensure longitude is in [0, 360)
    return distance, lon


def display_file_details(file_name, coordinate_system):
    """
    Displays the file details and includes the heliocentric position of the satellite
    (radial distance in AU and ecliptic longitude in degrees) for an event date.
    It also extracts and displays the day-of-year (ddoy) if present in the filename.
    """
    # Map the file prefix to mission names (adjust as needed)
    prefix_mapping = {
        "WI": "WIND",                      # WIND
        "PSP": "Parker Solar Probe",       # Parker Solar Probe
        "SO": "Solar Orbiter",             # Solar Orbiter
        "SOLO": "Solar Orbiter",           # Solar Orbiter
        "STA": "STEREO-A"                  # STEREO-A
    }
    file_parts = os.path.basename(file_name).split('_')
    
    # Assign the mission from the file prefix
    file_prefix = file_parts[0]
    full_name = prefix_mapping.get(file_prefix, "Unknown Mission")
    
    # Extract the file date: assumes the second part of the filename is the date in YYYYMMDD format
    try:
        file_date = pd.to_datetime(file_parts[1], format='%Y%m%d')
    except Exception:
        file_date = pd.Timestamp.now()
    file_year = file_date.year

    # Extract the day-of-year (ddoy) if present
    ddoy = None
    if len(file_parts) >= 3:
        try:
            ddoy = float(file_parts[2])
        except ValueError:
            ddoy = None

    # Extract the sampling interval: take the last part before the extension
    file_duration = file_parts[-1].split('.')[0]
    # Optionally, adjust if you need to convert, e.g. "hr" to "1hr"
    if file_duration.lower() == "hr":
        file_duration = "1hr"

    # -------------------------------------
    # Calculate the satellite's heliocentric position for the event
    # -------------------------------------
    # Use a fixed date for the event; you can also use file_date if desired
    obstime = parse_time("2025-04-07 00:00:00")
    try:
        # For this example, we use get_horizons_coord which works for several missions.
        coord = get_horizons_coord(full_name, obstime)
        distance, lon_ecliptic = get_position_data(full_name, coord, obstime)
    except Exception as e:
        st.warning(f"Could not obtain the satellite position for {full_name}: {e}")
        distance, lon_ecliptic = None, None

    # -------------------------------------
    # Display file details using Streamlit
    # -------------------------------------
    st.success(f"File {file_name} loaded successfully.")

    details_html = f"""
    <h3>File Details:</h3>
    <ul>
        <li><b>Mission:</b> {full_name}</li>
        <li><b>Date:</b> {file_date.strftime('%Y-%m-%d')}</li>
        <li><b>Day-of-Year (ddoy):</b> {ddoy}</li>
        <li><b>Coordinate System:</b> {coordinate_system}</li>
    </ul>
    """
        # <li><b>Sampling Interval:</b> {file_duration}</li>
        # <li><b>Satellite's Distance from the Sun (AU):</b> {np.round(distance, 3)}</li>
        # <li><b>Ecliptic Longitude (degrees):</b> {np.round(lon_ecliptic, 2)}</li>

    st.markdown(details_html, unsafe_allow_html=True)
        
    return full_name, file_date, file_duration, file_year, full_name


def display_data_info(data, coordinate_system=None):
    """
    Muestra una vista previa de los datos y estadísticas básicas.
    Si no se proporciona el sistema de coordenadas, se determina a partir de los datos.
    En caso de faltar columnas requeridas, se muestra una advertencia en lugar de detener la ejecución.
    """
    if coordinate_system is None:
        coordinate_system = identify_coordinate_system(data)
    st.write("### Data Preview:")
    if coordinate_system == "GSE":
        desired_columns = ['ddoy', 'B', 'Bx', 'By', 'Bz', 'Np', 'Vth', 'Vsw']
    elif coordinate_system == "RTN":
        desired_columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn', 'Np', 'Vth', 'Vsw']
    else:
        st.error("Unknown coordinate system.")
        return
    available_columns = [col for col in desired_columns if col in data.columns]
    data_display = data[available_columns]
    st.write(data_display.head())
    
    # Verificar columnas mínimas (para gráficos, se usan las componentes magnéticas)
    if coordinate_system == "GSE":
        required_columns = ['ddoy', 'B', 'Bx', 'By', 'Bz']
    elif coordinate_system == "RTN":
        required_columns = ['ddoy', 'B', 'Br', 'Bt', 'Bn']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"The following columns are missing and have been filled with NaN: {missing_columns}")
    st.write("### Basic Statistics of the Columns:")
    st.write(data_display.describe())