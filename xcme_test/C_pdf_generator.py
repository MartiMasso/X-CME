import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import io
from matplotlib import font_manager

# Enable LaTeX rendering and configure font settings
bold_font = font_manager.FontProperties(weight='bold')

plt.rcParams.update({
    "text.usetex": True,               # Enable LaTeX rendering
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Ensure consistent LaTeX style
    "axes.titlesize": 16,              # Set title size
    "axes.titleweight": "bold",        # Bold for titles
    "axes.labelweight": "bold"
})

def generate_pdf_report(data, file_name, file_date, file_duration, full_name, initial_date, final_date, filtered_data, coordinate_system):
    """
    Generate a PDF report containing data statistics and plots.
    """
    pdf_buffer = io.BytesIO()  # Create a buffer for the PDF

    with PdfPages(pdf_buffer) as pdf:
        # Global page size configuration
        page_size = (8.27, 11.69)  # A4 size in inches approximately

        # Page 1: Data configuration with header, date, and margin
        fig, ax = plt.subplots(figsize=page_size)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins

        # Header Title using LaTeX
        header_text = r"$\textbf{Flux\ Rope\ Analysis\ for\ Event\ XX}$"
        ax.axis("off")  # Disable the axes
        ax.text(
            0.5, 1.0, header_text, fontsize=14, va="top", ha="center"
        )

        # Add the date
        separation = 0.41 / 11.69  # Convert 0.5 cm to normalized coordinates
        report_date = r"Report Date: \texttt{" + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "}"
        ax.text(
            0.5, 1.0 - separation, report_date, fontsize=10, va="top", ha="center", color="gray"
        )

        # Calculate total duration
        duration = final_date - initial_date
        total_days = duration.days
        total_seconds = duration.seconds
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Ensure that file_date is a string
        if not isinstance(file_date, str):
            file_date = file_date.strftime('%Y-%m-%d')  # Convert to a string if it is not already
            
        # Formatted text for data details using LaTeX and string concatenation
        formatted_text_position = 1.0 - separation - (0.787 / 11.69)
        formatted_text = (
            r"$\textbf{Mission\ and\ Event\ Details:}$" + "\n\n"
            + r"\textbf{-- Mission:} " + full_name + "\n\n"
            + r"\textbf{-- Date:} " + file_date + "\n\n"
            + r"\textbf{-- Sampling Interval:} " + file_duration + "\n\n"
            + r"\textbf{-- Coordinate System:} " + coordinate_system + "\n\n\n"
            + r"$\textbf{Data\ Configuration\ Report:}$" + "\n\n"
            + r"\textbf{-- Selected start date:} " + initial_date.strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
            + r"\textbf{-- Selected end date:} " + final_date.strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
            + r"\textbf{-- Total duration of the event:} " + f"{total_days} days, {hours} hours, {minutes} minutes" + "\n\n"
            + r"\textbf{-- Number of data points:} " + str(len(filtered_data)) + "\n\n\n"
            + r"$\textbf{Basic\ Statistics\ of\ the\ Columns:}$" 
        )

        # Draw the formatted text below the date
        ax.text(
            0.1, formatted_text_position, formatted_text, fontsize=10, va="top", ha="left"
        )

        # Add basic statistics table with bold headers
        table_data = data.describe().round(4)
        table = plt.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            rowLabels=table_data.index,
            loc="center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.15, 0.22, 0.75, 0.3]  # [x0, y0, width, height]
        )

        # Apply bold styling to both column headers and row labels
        for key, cell in table.get_celld().items():
            if key[0] == 0 or key[1] == -1:  # Headers and row labels
                cell.set_text_props(fontweight='bold')

        # Apply styling adjustments
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(table_data.columns))))
        table.scale(1, 1.5)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Experimental Data Plots
        fig, ax = plt.subplots(4, 1, figsize=page_size, sharex=True)
        fig.subplots_adjust(left=0.15, right=0.85, top=0.86, bottom=0.25, hspace=0.5)

        # Add header to the second page
        fig.suptitle(r"$\textbf{Experimental\ Data\ Plots}$", fontsize=14, y=0.95)

        # Plot data with scatter plots and minor grid lines
        for i, (component, color) in enumerate(zip(['B', 'Bx', 'By', 'Bz'], ['blue', 'red', 'green', 'purple'])):
            ax[i].scatter(filtered_data['ddoy'], filtered_data[component], color=color, s=10, label=component)
            ax[i].set_title(rf"$\textbf{{Magnetic\ Field\ Component\ ({component})}}$", fontsize=10)
            ax[i].set_ylabel(rf"${component}$")
            ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)  # Minor grid enabled
            ax[i].minorticks_on()  # Turn on minor ticks
            ax[i].legend()

        ax[-1].set_xlabel(r"$ddoy$")

        pdf.savefig(fig)
        plt.close(fig)

    # Button to download the PDF
    st.download_button(
        label="Download Configuration Report",
        data=pdf_buffer.getvalue(),
        file_name="configuration_report.pdf",
        mime="application/pdf"
    )



