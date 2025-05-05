# metrics/visualisation.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, title, xlabel, ylabel, color='blue'):
    """
    Utility function to plot a histogram.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=20, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

def plot_metrics(detections_per_frame, tracks_per_frame):
    """
    This function will be responsible for plotting detection and tracking metrics as graphs.
    """
    st.title("📈 Inferred Metrics with visualisations")

    # --- Detection Metrics ---
    total_detections = sum(len(f) for f in detections_per_frame)
    avg_conf = np.mean([d[-1] for frame in detections_per_frame for d in frame]) if total_detections > 0 else 0
    st.subheader("Detection Metrics")
    st.metric("Total Detections", total_detections)
    st.metric("Avg Confidence", f"{avg_conf:.2f}")
    st.metric("Frames Processed", len(detections_per_frame))

    # Plot histogram for detection confidence
    conf_values = [d[-1] for frame in detections_per_frame for d in frame]
    if conf_values:
        st.subheader("Detection Confidence Histogram")
        plot_histogram(conf_values, "Detection Confidence Distribution", "Confidence", "Frequency", color='blue')

    # --- Tracking Metrics ---
    track_lengths = defaultdict(int)
    for frame in tracks_per_frame:
        for tid, *_ in frame:
            track_lengths[tid] += 1

    total_tracks = len(track_lengths)
    avg_track_length = np.mean(list(track_lengths.values())) if total_tracks > 0 else 0

    st.subheader("Tracking Metrics")
    st.metric("Unique Track IDs", total_tracks)
    st.metric("Avg Track Duration (frames)", f"{avg_track_length:.2f}")

    # Plot histogram for track lengths
    track_values = list(track_lengths.values())
    if track_values:
        st.subheader("Track Duration Histogram")
        plot_histogram(track_values, "Track Duration Distribution", "Duration (frames)", "Frequency", color='green')

def save_metrics_to_file(detections_per_frame, tracks_per_frame, file_path):
    """
    Save the detected metrics to a text file for review or export.
    """
    with open(file_path, 'w') as f:
        # Detection metrics
        total_detections = sum(len(f) for f in detections_per_frame)
        avg_conf = np.mean([d[-1] for frame in detections_per_frame for d in frame]) if total_detections > 0 else 0
        f.write(f"Total Detections: {total_detections}\n")
        f.write(f"Avg Confidence: {avg_conf:.2f}\n")
        f.write(f"Frames Processed: {len(detections_per_frame)}\n")

        # Tracking metrics
        track_lengths = defaultdict(int)
        for frame in tracks_per_frame:
            for tid, *_ in frame:
                track_lengths[tid] += 1
        total_tracks = len(track_lengths)
        avg_track_length = np.mean(list(track_lengths.values())) if total_tracks > 0 else 0
        f.write(f"Unique Track IDs: {total_tracks}\n")
        f.write(f"Avg Track Duration (frames): {avg_track_length:.2f}\n")

    st.success(f"Metrics saved to {file_path}")
