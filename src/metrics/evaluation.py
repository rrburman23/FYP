# metrics/evaluation.py

import streamlit as st
import numpy as np
from collections import defaultdict
from metrics.visualisation import plot_metrics, save_metrics_to_file

def display_metrics(detections_per_frame, tracks_per_frame):
    st.title("📈 Inferred Metrics")

    # Display the metrics and visualizations
    plot_metrics(detections_per_frame, tracks_per_frame)

    # Export metrics to file
    if st.button("📥 Save Metrics", use_container_width=True):
        file_path = "metrics_output.txt"
        save_metrics_to_file(detections_per_frame, tracks_per_frame, file_path)
        st.download_button("Download Metrics", data=open(file_path, 'r').read(), file_name="metrics_output.txt", mime="text/plain")
