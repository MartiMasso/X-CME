# xcme/B_plotting.py

# 0) Elegimos el backend Agg antes de cualquier import de pyplot
import matplotlib
matplotlib.use('Agg')

# 1) Import básico y rcParams
import matplotlib as mpl
mpl.rcParams['text.usetex']              = False
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_scientific'] = False
mpl.rcParams['axes.formatter.use_mathtext']   = False

# 2) Parche a TeXManager para que no intente generar DVI (y nunca falle)
import matplotlib.texmanager as _tm
_tm.TeXManager.get_text_width_height_descent = lambda self, tex, fontsize: (0, 0, 0)

# 3) Ya podemos importar pyplot y Streamlit sin riesgo
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from A_data import doy_2_datetime

def plot_data(data: pd.DataFrame, event_year: int):
    """
    Genera gráficos scatter y line para componentes del campo magnético.
    Detecta si usa GSE (Bx, By, Bz) o RTN (Br, Bt, Bn).
    """

    # — Selector de rango de ddoy —
    st.write("### Select Range of ddoy:")
    first_date, last_date = data['ddoy'].min(), data['ddoy'].max()
    ddoy_range = st.slider(
        "Select the range of ddoy for analysis:",
        min_value=float(first_date),
        max_value=float(last_date),
        value=(float(first_date), float(last_date))
    )

    # Cálculo de índices
    start_index = data[data['ddoy'] >= ddoy_range[0]].index[0]
    end_index   = data[data['ddoy'] <= ddoy_range[1]].index[-1]
    total_pts   = end_index - start_index + 1

    st.markdown(f"""
    <b>Start Index:</b> {start_index} &nbsp;&nbsp;
    <b>End Index:</b> {end_index} &nbsp;&nbsp;
    <b>Total Points:</b> {total_pts}
    """, unsafe_allow_html=True)

    # Fechas y duración
    ini = doy_2_datetime(ddoy_range[0], event_year)
    fin = doy_2_datetime(ddoy_range[1], event_year)
    delta = fin - ini
    days = delta.days
    hrs, rem = divmod(delta.seconds, 3600)
    mins, secs = divmod(rem, 60)

    st.write("### Selected Dates:")
    st.markdown(f"""
    <ul>
      <li><b>Start:</b> {ini:%Y-%m-%d} {ini:%H:%M:%S}</li>
      <li><b>End:</b>   {fin:%Y-%m-%d} {fin:%H:%M:%S}</li>
      <li><b>Total Duration:</b> {days} days, {hrs} hours, {mins} minutes</li>
    </ul>
    """, unsafe_allow_html=True)

    # Filtrar datos
    filtered = data[(data['ddoy'] >= ddoy_range[0]) & (data['ddoy'] <= ddoy_range[1])]

    # Detectar sistema de coordenadas
    if 'Bx' in filtered.columns:
        comp1, comp2, comp3 = 'Bx', 'By', 'Bz'
    elif 'Br' in filtered.columns:
        comp1, comp2, comp3 = 'Br', 'Bt', 'Bn'
    else:
        st.error("No magnetic field component columns found.")
        return

    # — Scatter plots —
    st.write("### Magnetic Field Components (Scatter):")
    fig_s, axes_s = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig_s.subplots_adjust(left=0.18, right=0.88, top=0.86, bottom=0.2, hspace=0.4)

    axes_s[0].scatter(filtered['ddoy'], filtered['B'], s=10, label='B')
    axes_s[0].set_title('Magnetic Field Intensity (B)')
    axes_s[0].set_ylabel('B'); axes_s[0].grid(True); axes_s[0].legend()

    axes_s[1].scatter(filtered['ddoy'], filtered[comp1], s=10, label=comp1)
    axes_s[1].set_title(f'Component {comp1}'); axes_s[1].set_ylabel(comp1)
    axes_s[1].grid(True); axes_s[1].legend()

    axes_s[2].scatter(filtered['ddoy'], filtered[comp2], s=10, label=comp2)
    axes_s[2].set_title(f'Component {comp2}'); axes_s[2].set_ylabel(comp2)
    axes_s[2].grid(True); axes_s[2].legend()

    axes_s[3].scatter(filtered['ddoy'], filtered[comp3], s=10, label=comp3)
    axes_s[3].set_title(f'Component {comp3}'); axes_s[3].set_ylabel(comp3)
    axes_s[3].set_xlabel('ddoy'); axes_s[3].grid(True); axes_s[3].legend()

    st.pyplot(fig_s)
    plt.close(fig_s)

    # — Line plots —
    st.write("### Magnetic Field Components (Line):")
    fig_l, axes_l = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig_l.subplots_adjust(left=0.18, right=0.88, top=0.86, bottom=0.2, hspace=0.4)

    axes_l[0].plot(filtered['ddoy'], filtered['B'], label='B')
    axes_l[0].set_title('Magnetic Field Intensity (B)')
    axes_l[0].set_ylabel('B'); axes_l[0].grid(True); axes_l[0].legend()

    axes_l[1].plot(filtered['ddoy'], filtered[comp1], label=comp1)
    axes_l[1].set_title(f'Component {comp1}'); axes_l[1].set_ylabel(comp1)
    axes_l[1].grid(True); axes_l[1].legend()

    axes_l[2].plot(filtered['ddoy'], filtered[comp2], label=comp2)
    axes_l[2].set_title(f'Component {comp2}'); axes_l[2].set_ylabel(comp2)
    axes_l[2].grid(True); axes_l[2].legend()

    axes_l[3].plot(filtered['ddoy'], filtered[comp3], label=comp3)
    axes_l[3].set_title(f'Component {comp3}'); axes_l[3].set_ylabel(comp3)
    axes_l[3].set_xlabel('ddoy'); axes_l[3].grid(True); axes_l[3].legend()

    st.pyplot(fig_l)
    plt.close(fig_l)

    return ini, fin, filtered, days, hrs, mins, start_index, end_index
