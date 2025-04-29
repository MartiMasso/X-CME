import matplotlib as mpl
mpl.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from A_data import doy_2_datetime

def plot_data(data, event_year):
    """
    Generate both scatter and line plots for magnetic field components with range selection.
    Detects whether the data is in GSE (Bx, By, Bz) or RTN (Br, Bt, Bn) and plots accordingly.
    """
    # Mostrar el rango de ddoy y los índices seleccionados
    st.write("### Select Range of ddoy:")
    first_date, last_date = data['ddoy'].min(), data['ddoy'].max()
    ddoy_range = st.slider(
        "Select the range of ddoy for analysis:",
        min_value=float(first_date),
        max_value=float(last_date),
        value=(float(first_date), float(last_date))
    )

    # Calcular los índices de inicio y fin usando la selección
    start_index = data[data['ddoy'] >= ddoy_range[0]].index[0]
    end_index = data[data['ddoy'] <= ddoy_range[1]].index[-1]
    total_points = end_index - start_index + 1

    st.markdown(f"""
    <b>Start Index:</b> {start_index} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    <b>End Index:</b> {end_index} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    <b>Total Points:</b> {total_points}
    """, unsafe_allow_html=True)

    # Calcular las fechas inicial y final
    initial_date = doy_2_datetime(ddoy_range[0], event_year)
    final_date = doy_2_datetime(ddoy_range[1], event_year)

    # Calcular la duración total
    duration = final_date - initial_date
    total_days = duration.days
    total_seconds = duration.seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    st.write("### Selected Dates:")
    st.markdown(f"""
    <ul>
        <li><b>Start Date:</b> {initial_date.strftime('%Y-%m-%d')} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Time:</b> {initial_date.strftime('%H:%M:%S')}</li>
        <li><b>End Date:</b> {final_date.strftime('%Y-%m-%d')} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Time:</b> {final_date.strftime('%H:%M:%S')}</li>
        <li><b>Total Duration:</b> {total_days} days, {hours} hours, {minutes} minutes</li>
    </ul>
    """, unsafe_allow_html=True)

    # Filtrar los datos según el rango seleccionado
    filtered_data = data[(data['ddoy'] >= ddoy_range[0]) & (data['ddoy'] <= ddoy_range[1])]

    # Determinar qué columnas magnéticas utilizar según el sistema de coordenadas
    if 'Bx' in filtered_data.columns:
        # Sistema GSE
        comp1, comp2, comp3 = 'Bx', 'By', 'Bz'
    elif 'Br' in filtered_data.columns:
        # Sistema RTN
        comp1, comp2, comp3 = 'Br', 'Bt', 'Bn'
    else:
        st.error("No magnetic field component columns found.")
        return None

    st.write("### Magnetic Field Components (Scatter):")
    fig_scatter, ax_scatter = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig_scatter.subplots_adjust(left=0.18, right=0.88, top=0.86, bottom=0.2, hspace=0.4)

    # Primer subplot: Magnitud del campo B
    ax_scatter[0].scatter(filtered_data['ddoy'], filtered_data['B'], color='blue', s=10, label='B')
    ax_scatter[0].set_title('Magnetic Field Intensity (B)')
    ax_scatter[0].set_ylabel('B')
    ax_scatter[0].grid(True)
    ax_scatter[0].legend()

    # Segundo subplot: Primer componente
    ax_scatter[1].scatter(filtered_data['ddoy'], filtered_data[comp1], color='red', s=10, label=comp1)
    ax_scatter[1].set_title(f'Magnetic Field Component ({comp1})')
    ax_scatter[1].set_ylabel(comp1)
    ax_scatter[1].grid(True)
    ax_scatter[1].legend()

    # Tercer subplot: Segundo componente
    ax_scatter[2].scatter(filtered_data['ddoy'], filtered_data[comp2], color='green', s=10, label=comp2)
    ax_scatter[2].set_title(f'Magnetic Field Component ({comp2})')
    ax_scatter[2].set_ylabel(comp2)
    ax_scatter[2].grid(True)
    ax_scatter[2].legend()

    # Cuarto subplot: Tercer componente
    ax_scatter[3].scatter(filtered_data['ddoy'], filtered_data[comp3], color='purple', s=10, label=comp3)
    ax_scatter[3].set_title(f'Magnetic Field Component ({comp3})')
    ax_scatter[3].set_xlabel('ddoy')
    ax_scatter[3].set_ylabel(comp3)
    ax_scatter[3].grid(True)
    ax_scatter[3].legend()

    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

    st.write("### Magnetic Field Components (Line):")
    fig_line, ax_line = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig_line.subplots_adjust(left=0.18, right=0.88, top=0.86, bottom=0.2, hspace=0.4)

    ax_line[0].plot(filtered_data['ddoy'], filtered_data['B'], color='blue', label='B')
    ax_line[0].set_title('Magnetic Field Intensity (B)')
    ax_line[0].set_ylabel('B')
    ax_line[0].grid(True)
    ax_line[0].legend()

    ax_line[1].plot(filtered_data['ddoy'], filtered_data[comp1], color='red', label=comp1)
    ax_line[1].set_title(f'Magnetic Field Component ({comp1})')
    ax_line[1].set_ylabel(comp1)
    ax_line[1].grid(True)
    ax_line[1].legend()

    ax_line[2].plot(filtered_data['ddoy'], filtered_data[comp2], color='green', label=comp2)
    ax_line[2].set_title(f'Magnetic Field Component ({comp2})')
    ax_line[2].set_ylabel(comp2)
    ax_line[2].grid(True)
    ax_line[2].legend()

    ax_line[3].plot(filtered_data['ddoy'], filtered_data[comp3], color='purple', label=comp3)
    ax_line[3].set_title(f'Magnetic Field Component ({comp3})')
    ax_line[3].set_xlabel('ddoy')
    ax_line[3].set_ylabel(comp3)
    ax_line[3].grid(True)
    ax_line[3].legend()

    st.pyplot(fig_line)
    plt.close(fig_line)

    return initial_date, final_date, filtered_data, total_days, hours, minutes, start_index, end_index
