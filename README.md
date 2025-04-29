# X-CME

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15306061.svg)](https://doi.org/10.5281/zenodo.15306061)

**Fitting app for interplanetary Coronal Mass Ejections (ICMEs)**  
Developed by Martí Massó Moreno at NASA Goddard

---

## 🚀 Overview

X-CME is a Python/Streamlit application for interactive, reproducible analysis of Coronal Mass Ejections (CMEs) and their embedded magnetic flux ropes using in‑situ spacecraft data. Key features include:

- **Flux Rope Analysis** with two physical variants (purely radial; radial + azimuthal)  
- **Real-time parameter sliders** for geometry, field configuration, and propagation speed  
- **Publication‑quality plots** of magnetic field and current density distributions  
- **Interactive 3D visualizations** of flux ropes and CME propagation paths

---

## 🔧 Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/X-CME/X-CME.git
   cd X-CME
   ```
2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Local

```bash
streamlit run app/streamlit_app.py
```

Open your browser at <http://localhost:8501> to interact with X-CME locally.

### Streamlit Community Cloud

1. Push your code to GitHub and ensure you have a tagged **release** (v0.1.5).  
2. Go to <https://share.streamlit.io>, connect your GitHub account, select **X-CME/X-CME**, branch `main`, and file `app/streamlit_app.py`.  
3. Click **Deploy** and share the generated URL (e.g., `https://share.streamlit.io/X-CME/X-CME/main/app/streamlit_app.py`).

---

## 📑 How to cite

If you use X-CME in your research, please cite it as follows:

```bibtex
@software{MassoMoreno2025,
  title    = {{X-CME: Fitting App for ICME’s}},
  author   = {Massó Moreno, Martí},
  year     = {2025},
  version  = {v0.1.5},
  doi      = {10.5281/zenodo.15306061},
  url      = {https://github.com/X-CME/X-CME}
}
```

Metadata is also available in [`CITATION.cff`](CITATION.cff).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

