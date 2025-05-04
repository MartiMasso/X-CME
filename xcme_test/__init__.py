## FYI: This is not being used anywhere -- A.A.N.

# -------------------- Parche global contra LaTeX en matplotlib --------------------
import matplotlib

# Usar Agg para renderizar sin GUI
matplotlib.use('Agg')

import matplotlib as mpl

# Desactivar cualquier uso de LaTeX externo o mathtext avanzado
mpl.rcParams['text.usetex']                 = False
mpl.rcParams['axes.formatter.useoffset']    = False
mpl.rcParams['axes.formatter.use_scientific']= False
mpl.rcParams['axes.formatter.use_mathtext'] = False

# Evitar que TeXManager intente llamar a latex/dvipng
import matplotlib.texmanager as _tm
_tm.TeXManager.get_text_width_height_descent = lambda self, tex, fontsize: (0, 0, 0)
# ------------------------------------------------------------------------------------
