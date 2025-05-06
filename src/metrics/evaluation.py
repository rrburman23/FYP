# metrics/evaluation.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import psutil
import logging
import json
import hashlib
from typing import List, Dict, Any, Union

# Configure logging for debugging and error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for plots with size limit
if "plot_cache" not in st.session_state:
    st.session_state.plot_cache = {}
MAX_CACHE_SIZE = 10  # Limit cache to 10 plots

def clear_plot_cache() -> None:
    """
    Clear the plot cache to free memory.
    """
    st.session_state.plot_cache.clear()
    logger.debug("Cleared plot cache")

def plot_histogram(data: List[float], title: str, xlabel: str, ylabel: str, color: str = 'blue') -> None:
    """
    Plot a histogram of the given data and display it in Streamlit, using cached plots if available.

    Args:
        data (List[float]): Data to plot (e.g., track durations).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color (str): Histogram color (default: 'blue').
    """
    if not data or len(data) < 10:  # Skip small datasets
        logger.debug(f"Skipping histogram due to insufficient data: {title}")
        st.warning(f"Insufficient data for {title}")
        return

    # Generate cache key based on data hash
    data_hash = hashlib.md5(str(data).encode()).hexdigest()
    cache_key = f"histogram_{title}_{data_hash}"
    
    # Check cache
    if cache_key in st.session_state.plot_cache:
        logger.debug(f"Using cached histogram: {title}")
        st.pyplot(st.session_state.plot_cache[cache_key])
        return

    # Create histogram
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=min(20, len(data)), color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Cache and display plot
        if len(st.session_state.plot_cache) >= MAX_CACHE_SIZE:
            st.session_state.plot_cache.pop(next(iter(st.session_state.plot_cache)))  # Remove oldest
        st.session_state.plot_cache[cache_key] = fig
        st.pyplot(fig)
    finally:
        plt.close(fig)  # Ensure figure is closed
        logger.debug(f"Created and closed histogram: {title}")

def plot_fps(fps_values: List[float]) -> None:
    """
    Plot FPS over time as a line graph and display it in Streamlit, using cached plots if available.

    Args:
        fps_values (List[float]): List of FPS values per frame.
    """
    if not fps_values or len(fps_values) < 10:  # Skip small datasets
        logger.debug("Skipping FPS plot due to insufficient data")
        st.warning("Insufficient FPS data")
        return

    # Generate cache key based on data hash
    data_hash = hashlib.md5(str(fps_values).encode()).hexdigest()
    cache_key = f"fps_{data_hash}"
    
    # Check cache
    if cache_key in st.session_state.plot_cache:
        logger.debug(f"Using cached FPS plot")
        st.pyplot(st.session_state.plot_cache[cache_key])
        return

    # Create line plot
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fps_values[:1000], label="FPS", color="red")  # Limit to 1000 points
        ax.set_xlabel("Frame")
        ax.set_ylabel("FPS")
        ax.set_title("FPS Over Time")
        ax.legend()
        
        # Cache and display plot
        if len(st.session_state.plot_cache) >= MAX_CACHE_SIZE:
            st.session_state.plot_cache.pop(next(iter(st.session_state.plot_cache)))  # Remove oldest
        st.session_state.plot_cache[cache_key] = fig
        st.pyplot(fig)
    finally:
        plt.close(fig)  # Ensure figure is closed
        logger.debug(f"Created and closed FPS plot")

def calculate_id_switches(tracks_per_frame: List[List[Dict[str, Any]]]) -> int:
    """
    Calculate the number of ID switches by comparing track IDs across frames, accounting for natural terminations.

    Args:
        tracks_per_frame (List[List[Dict[str, Any]]]): List of frames, each containing track dictionaries.

    Returns:
        int: Number of ID switches.
    """
    id_switches = 0
    previous_ids = set()
    previous_boxes = {}  # Track ID -> bounding box for overlap check

    for frame in tracks_per_frame:
        current_ids = set()
        current_boxes = {}
        for track in frame:
            if isinstance(track, dict) and "track_id" in track:
                current_ids.add(track["track_id"])
                if "bbox" in track:  # Assuming bbox is [x1, y1, x2, y2]
                    current_boxes[track["track_id"]] = track["bbox"]

        # Check for ID switches (exclude natural terminations)
        for prev_id in previous_ids - current_ids:
            # Skip if no overlap with any current track (likely exited frame)
            if prev_id in previous_boxes:
                prev_box = previous_boxes[prev_id]
                has_overlap = False
                for curr_box in current_boxes.values():
                    if _calculate_iou(prev_box, curr_box) > 0.1:  # IoU threshold
                        has_overlap = True
                        break
                if has_overlap:
                    id_switches += 1
            else:
                id_switches += 1  # Fallback if no bbox

        previous_ids = current_ids
        previous_boxes = current_boxes

    logger.debug(f"Calculated {id_switches} ID switches")
    return id_switches

def calculate_fragmentation(tracks_per_frame: List[List[Dict[str, Any]]]) -> int:
    """
    Calculate fragmentation by counting new track IDs per frame, accounting for natural starts.

    Args:
        tracks_per_frame (List[List[Dict[str, Any]]]): List of frames, each containing track dictionaries.

    Returns:
        int: Fragmentation count.
    """
    fragmentation = 0
    previous_ids = set()
    previous_boxes = {}  # Track ID -> bounding box for overlap check

    for frame in tracks_per_frame:
        current_ids = set()
        current_boxes = {}
        for track in frame:
            if isinstance(track, dict) and "track_id" in track:
                current_ids.add(track["track_id"])
                if "bbox" in track:
                    current_boxes[track["track_id"]] = track["bbox"]

        # Count new IDs, excluding those with no overlap (likely new objects)
        for curr_id in current_ids - previous_ids:
            if curr_id in current_boxes:
                curr_box = current_boxes[curr_id]
                has_overlap = False
                for prev_box in previous_boxes.values():
                    if _calculate_iou(curr_box, prev_box) > 0.1:  # IoU threshold
                        has_overlap = True
                        break
                if has_overlap:
                    fragmentation += 1
            else:
                fragmentation += 1  # Fallback if no bbox

        previous_ids = current_ids
        previous_boxes = current_boxes

    logger.debug(f"Calculated {fragmentation} fragmentation events")
    return fragmentation

def _calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1 (List[float]): Bounding box [x1, y1, x2, y2].
        box2 (List[float]): Bounding box [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_tracking_metrics(
    detections_per_frame: List[List[List[Union[int, float]]]],
    tracks_per_frame: List[List[Dict[str, Any]]],
    start_idx: int = 0
) -> tuple:
    """
    Calculate tracking metrics from detections and tracks, starting from a given index.

    Args:
        detections_per_frame (List[List[List[Union[int, float]]]]): Detections per frame ([x1, y1, x2, y2, conf]).
        tracks_per_frame (List[List[Dict[str, Any]]]): Tracks per frame (dicts with track_id).
        start_idx (int): Index to start processing from (for incremental updates).

    Returns:
        tuple: (total_detections, total_tracks, avg_track_length, id_switches, fragmentation, track_lengths)
    """
    # Validate detections
    if not detections_per_frame or not all(isinstance(f, list) for f in detections_per_frame):
        logger.warning("Invalid or empty detections_per_frame")
        total_detections = 0
    else:
        # Validate detection format
        total_detections = 0
        for frame in detections_per_frame[start_idx:]:
            for det in frame:
                if not (isinstance(det, list) and len(det) == 5):
                    logger.warning("Invalid detection format: Expected [x1, y1, x2, y2, confidence]")
                    continue
                total_detections += 1

    if not tracks_per_frame or not all(isinstance(f, list) for f in tracks_per_frame):
        logger.warning("Invalid or empty tracks_per_frame")
        return total_detections, 0, 0.0, 0, 0, defaultdict(int)

    # Calculate track lengths
    track_lengths = defaultdict(int)
    for frame in tracks_per_frame[start_idx:]:
        for track in frame:
            if isinstance(track, dict) and "track_id" in track:
                track_lengths[track["track_id"]] += 1

    # Compute metrics
    total_tracks = len(track_lengths)
    avg_track_length = sum(track_lengths.values()) / total_tracks if total_tracks > 0 else 0.0
    id_switches = calculate_id_switches(tracks_per_frame[start_idx:])
    fragmentation = calculate_fragmentation(tracks_per_frame[start_idx:])

    logger.info(f"Tracking metrics: {total_detections} detections, {total_tracks} tracks, {avg_track_length:.2f} avg length")
    return total_detections, total_tracks, avg_track_length, id_switches, fragmentation, track_lengths

def track_fps_and_memory(fps_values: List[float]) -> tuple:
    """
    Calculate average FPS (using rolling average) and memory usage.

    Args:
        fps_values (List[float]): List of FPS values per frame.

    Returns:
        tuple: (avg_fps, memory_usage in MB)
    """
    if not fps_values:
        logger.warning("No FPS values provided")
        return 0.0, 0.0

    # Calculate rolling average FPS (window size = 10)
    valid_fps = [f for f in fps_values if 0 < f < 1000]
    if not valid_fps:
        logger.warning("No valid FPS values after filtering")
        return 0.0, 0.0
    window_size = min(10, len(valid_fps))
    rolling_avg = np.convolve(valid_fps, np.ones(window_size)/window_size, mode='valid')
    avg_fps = np.mean(rolling_avg) if rolling_avg.size > 0 else 0.0

    # Measure memory usage
    try:
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # In MB
    except Exception as e:
        logger.error(f"Failed to measure memory usage: {str(e)}")
        memory_usage = 0.0

    logger.info(f"Computed FPS: {avg_fps:.2f}, Memory: {memory_usage:.2f} MB")
    return avg_fps, memory_usage

def display_metrics(
    detections_per_frame: List[List[List[Union[int, float]]]],
    tracks_per_frame: List[List[Dict[str, Any]]],
    fps_values: List[float] | None = None,
    start_idx: int = 0
) -> None:
    """
    Display tracking and computational metrics in Streamlit.

    Args:
        detections_per_frame (List[List[List[Union[int, float]]]]): Detections per frame.
        tracks_per_frame (List[List[Dict[str, Any]]]): Tracks per frame.
        fps_values (List[float] | None): FPS values per frame (optional).
        start_idx (int): Index to start processing from (for incremental updates).
    """
    with st.container():
        st.subheader("📈 Inferred Metrics", anchor=False)

        # Calculate and display tracking metrics
        total_detections, total_tracks, avg_track_length, id_switches, fragmentation, track_lengths = (
            calculate_tracking_metrics(detections_per_frame, tracks_per_frame, start_idx)
        )
        
        with st.expander("Tracking Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Detections", total_detections)
                st.metric("Unique Tracks", total_tracks)
                st.metric("Avg Track Duration (frames)", f"{avg_track_length:.2f}")
            with col2:
                st.metric("ID Switches", id_switches)
                st.metric("Fragmentation", fragmentation)

        # Plot track duration histogram
        track_values = list(track_lengths.values())
        if track_values and st.session_state.show_plots:
            with st.expander("Track Duration Histogram", expanded=False):
                plot_histogram(track_values, "Track Duration Distribution", "Duration (frames)", "Frequency", color='green')

        # Display computational efficiency
        if fps_values:
            fps, memory_usage = track_fps_and_memory(fps_values)
            with st.expander("Computational Efficiency", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average FPS", f"{fps:.2f}")
                with col2:
                    st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
                if st.session_state.show_plots:
                    plot_fps(fps_values)

def save_metrics_to_file(
    detections_per_frame: List[List[List[Union[int, float]]]],
    tracks_per_frame: List[List[Dict[str, Any]]],
    fps_values: List[float],
    file_path: str,
    format: str = "txt"
) -> None:
    """
    Save metrics to a file in text or JSON format.

    Args:
        detections_per_frame (List[List[List[Union[int, float]]]]): Detections per frame.
        tracks_per_frame (List[List[Dict[str, Any]]]): Tracks per frame.
        fps_values (List[float]): FPS values per frame.
        file_path (str): Path to save the metrics file.
        format (str): File format ('txt' or 'json', default: 'txt').

    Raises:
        IOError: If file writing fails.
        ValueError: If invalid format is specified.
    """
    # Calculate metrics
    total_detections, total_tracks, avg_track_length, id_switches, fragmentation, _ = (
        calculate_tracking_metrics(detections_per_frame, tracks_per_frame)
    )
    avg_fps, memory_usage = track_fps_and_memory(fps_values)

    # Prepare metrics dictionary
    metrics = {
        "total_detections": total_detections,
        "unique_tracks": total_tracks,
        "avg_track_length": avg_track_length,
        "id_switches": id_switches,
        "fragmentation": fragmentation,
        "average_fps": avg_fps,
        "memory_usage_mb": memory_usage
    }

    # Write to file
    try:
        if format == "txt":
            with open(file_path, 'w') as f:
                f.write(f"Total Detections: {total_detections}\n")
                f.write(f"Unique Tracks: {total_tracks}\n")
                f.write(f"Avg Track Duration (frames): {avg_track_length:.2f}\n")
                f.write(f"ID Switches: {id_switches}\n")
                f.write(f"Fragmentation: {fragmentation}\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n")
                f.write(f"Memory Usage (MB): {memory_usage:.2f}\n")
        elif format == "json":
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info(f"Metrics saved to {file_path} in {format} format")
    except IOError as e:
        logger.error(f"Failed to save metrics to {file_path}: {str(e)}")
        raise IOError(f"Failed to save metrics: {str(e)}")