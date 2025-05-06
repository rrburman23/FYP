import os
import time
import torch
import streamlit as st
from pathlib import Path
import cv2
import numpy as np
import logging
from datetime import datetime
import tempfile
import json
import importlib
import shutil
import subprocess
import zipfile
from typing import Dict, List, Any

from util.visualise import draw_detections, draw_tracks

# Suppress console logging for all modules to reduce clutter
logging.getLogger().handlers = []
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('util.paths').handlers = []
logging.getLogger('util.paths').setLevel(logging.WARNING)

# Configure app-specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []

# Configure Streamlit page settings for a wide layout and expanded sidebar
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable Streamlit's file watcher to prevent unnecessary reruns
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
torch.classes.__path__ = []

# Import custom modules for paths, processing, and metrics
try:
    from util.paths import DATA_DIR, OUTPUT_DIR, TEMP_DIR, load_config
    from processing.run_pair import run_pair
    from metrics.evaluation import display_metrics, clear_plot_cache, save_metrics_to_file
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Failed to import modules: {e}")
    st.stop()

# Ensure TEMP_DIR exists for storing temporary files
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Load configuration from config/default.yml
try:
    cfg = load_config()
    if not cfg or "detectors" not in cfg or "trackers" not in cfg:
        raise ValueError("Invalid configuration: 'detectors' or 'trackers' section missing")
    if f"{datetime.now()}: Loaded configuration from config/default.yml" not in st.session_state.get('logs', []):
        st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Loaded configuration from config/default.yml")
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Configuration loading failed: {e}")
    st.stop()

# Check for FFmpeg availability (bundled or system)
ffmpeg_path = Path("bin/ffmpeg.exe")
ffmpeg_available = False
if ffmpeg_path.exists():
    try:
        subprocess.run([str(ffmpeg_path), "-version"], capture_output=True, check=True)
        ffmpeg_available = True
        st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Found bundled FFmpeg at {ffmpeg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Bundled FFmpeg check failed: {e}")
else:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_available = True
        st.session_state.setdefault('logs', []).append(f"{datetime.now()}: Found system FFmpeg")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        st.warning(
            f"""
            FFmpeg is not found in project directory ({ffmpeg_path}) or system PATH. Video processing is disabled.
            You can still process image sequences (ZIP of JPEG/PNG files).
            To enable video support, place `ffmpeg.exe` in `{ffmpeg_path}`:
            1. Download from https://www.gyan.dev/ffmpeg/builds/ (e.g., ffmpeg-release-essentials.zip).
            2. Extract `bin/ffmpeg.exe` to `{ffmpeg_path}`.
            3. Verify with: `bin\\ffmpeg.exe -version`.
            """
        )
        st.session_state.setdefault('logs', []).append(f"{datetime.now()}: FFmpeg not found: {e}")

# Define supported detectors and trackers
DETECTORS = ["YOLOv3", "YOLOv5", "SSD", "FasterRCNN"]
TRACKERS = ["DeepSORT", "SORT", "KCF", "MOSSE", "MedianFlow"]

# Initialize session state with default values
_defaults = dict(
    page="📜 Instructions",
    queue=[],
    run=False,
    video=None,
    video_path=None,
    image_sequence=None,
    preview_holder=None,
    stop_processing=False,
    job_index=0,
    logs=[],
    job_metrics={},
    show_plots=True,
    preview_skip_frames=cfg["processing"].get("default_skip_frames", 10),
    metrics_start_idx=0,
    plot_cache={},
)

for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# Sidebar UI for video or image sequence upload
with st.sidebar:
    st.header("Configuration")

    # File uploader for videos or ZIP of images
    upload_type = st.radio("Upload Type", ["Video (MP4/MOV/AVI)", "Image Sequence (ZIP of JPEG/PNG)"], index=0)
    if upload_type == "Video (MP4/MOV/AVI)":
        up = st.file_uploader("Upload Video", type=("mp4", "mov", "avi"))
        if up:
            if up.size > 200 * 1024 * 1024:
                st.error("Video file is too large (max 200MB)")
                st.session_state.logs.append(f"{datetime.now()}: Video file too large: {up.name}")
            elif up.size < 1024:
                st.error("Video file is too small or empty")
                st.session_state.logs.append(f"{datetime.now()}: Empty or invalid video file: {up.name}")
            else:
                try:
                    temp_video = TEMP_DIR / up.name
                    with open(temp_video, "wb") as f:
                        f.write(up.read())
                    if not ffmpeg_available:
                        st.error(f"FFmpeg is required for video processing. Place `ffmpeg.exe` in `{ffmpeg_path}` or upload an image sequence.")
                        os.unlink(temp_video)
                    else:
                        cap = cv2.VideoCapture(str(temp_video))
                        if not cap.isOpened():
                            st.error(
                                """
                                Invalid video file. The file may be corrupted or missing metadata (moov atom not found).
                                Try re-encoding with FFmpeg:
                                ```bash
                                bin\\ffmpeg.exe -i your_video.mp4 -c:v copy -c:a copy -movflags +faststart fixed_video.mp4
                                ```
                                """
                            )
                            st.session_state.logs.append(f"{datetime.now()}: Invalid video file: {up.name}")
                            os.unlink(temp_video)
                        else:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width <= 0 or height <= 0:
                                st.error("Invalid video dimensions. Please upload a valid video.")
                                st.session_state.logs.append(f"{datetime.now()}: Invalid video dimensions: {up.name}")
                                os.unlink(temp_video)
                            else:
                                st.session_state.video = up
                                st.session_state.video_path = temp_video
                                st.session_state.image_sequence = None
                                st.session_state.logs.append(f"{datetime.now()}: Uploaded and validated video: {up.name}")
                        cap.release()
                except Exception as e:
                    st.error(f"Failed to process uploaded video: {e}")
                    st.session_state.logs.append(f"{datetime.now()}: Video upload failed: {e}")
                    if temp_video.exists():
                        os.unlink(temp_video)
    else:
        up = st.file_uploader("Upload Image Sequence (ZIP)", type="zip")
        if up:
            try:
                temp_zip = TEMP_DIR / up.name
                with open(temp_zip, "wb") as f:
                    f.write(up.read())
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(TEMP_DIR / "images")
                image_files = sorted(
                    [f for f in (TEMP_DIR / "images").glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
                    key=lambda x: x.name
                )
                if not image_files:
                    st.error("No valid images (JPEG/PNG) found in the ZIP file.")
                    st.session_state.logs.append(f"{datetime.now()}: Invalid image sequence: No JPEG/PNG files")
                    shutil.rmtree(TEMP_DIR / "images", ignore_errors=True)
                    os.unlink(temp_zip)
                else:
                    st.session_state.video = None
                    st.session_state.video_path = None
                    st.session_state.image_sequence = image_files
                    st.session_state.logs.append(f"{datetime.now()}: Uploaded image sequence with {len(image_files)} images")
            except Exception as e:
                st.error(f"Failed to process uploaded ZIP: {e}")
                st.session_state.logs.append(f"{datetime.now()}: ZIP upload failed: {e}")
                if (TEMP_DIR / "images").exists():
                    shutil.rmtree(TEMP_DIR / "images", ignore_errors=True)
                if temp_zip.exists():
                    os.unlink(temp_zip)

    # Detector and tracker selection without defaults
    det_sel = st.multiselect("Select Detectors", DETECTORS)
    trk_sel = st.multiselect("Select Trackers", TRACKERS)

    # Preview update frequency slider
    st.markdown("**Preview Update Frequency**")
    st.markdown("Controls how often the preview updates (in frames).")
    st.session_state.preview_skip_frames = st.slider(
        "Preview Update Frequency (frames)",
        1,
        100,
        st.session_state.preview_skip_frames,
    )

    # Add jobs to queue with validation
    if st.button("➕ Add to Queue", use_container_width=True):
        if not (st.session_state.video or st.session_state.image_sequence):
            st.error("Please upload a valid video or image sequence first")
            st.session_state.logs.append(f"{datetime.now()}: Queue addition failed: No valid input uploaded")
        elif not det_sel or not trk_sel:
            st.error("Please select at least one detector and tracker")
            st.session_state.logs.append(f"{datetime.now()}: Queue addition failed: No detectors or trackers selected")
        else:
            for d in det_sel:
                for t in trk_sel:
                    job = {"detector": d, "tracker": t}
                    if job not in st.session_state.queue:
                        st.session_state.queue.append(job)
                        st.session_state.logs.append(f"{datetime.now()}: Added job {d} + {t} to queue")
            st.session_state.page = "📋 Queue"
            st.rerun()

    # Run processing with validation
    if st.button("▶️ Run Processing", use_container_width=True):
        if not (st.session_state.video or st.session_state.image_sequence) or not st.session_state.queue:
            st.error("Upload a video or image sequence and add jobs to the queue first")
            st.session_state.logs.append(f"{datetime.now()}: Processing failed: Missing input or queue")
        elif st.session_state.video and not ffmpeg_available:
            st.error(f"FFmpeg is required for video processing. Place `ffmpeg.exe` in `{ffmpeg_path}` or use an image sequence.")
            st.session_state.logs.append(f"{datetime.now()}: Processing failed: FFmpeg required for video")
        else:
            st.session_state.run = True
            st.session_state.stop_processing = False
            st.session_state.preview_holder = None
            st.session_state.page = "👁 Preview"
            st.session_state.logs.append(f"{datetime.now()}: Initiated processing")
            st.rerun()

# Custom CSS for queue table and buttons
st.markdown(
    """
    <style>
    .queue-table {
        width: 100%;
        margin: 1rem 0;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .queue-table th, .queue-table td {
        padding: 12px 16px;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
        font-size: 0.95rem;
        color: #1f2937;
        min-height: 48px;
        line-height: 1.5;
    }
    .queue-table th {
        background: #2563eb; /* Flat blue, non-shiny */
        color: #ffffff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #93c5fd;
    }
    .queue-table th:nth-child(1) {
        width: 10%;
    }
    .queue-table th:nth-child(2) {
        width: 35%;
    }
    .queue-table th:nth-child(3) {
        width: 35%;
    }
    .queue-table th:nth-child(4) {
        width: 20%;
    }
    .queue-table td {
        background: #fafafa;
        transition: background 0.3s ease;
    }
    .queue-table td:nth-child(1) {
        width: 10%;
        font-weight: 600;
        color: #1e40af;
    }
    .queue-table td:nth-child(2) {
        width: 35%;
    }
    .queue-table td:nth-child(3) {
        width: 35%;
    }
    .queue-table td:nth-child(4) {
        width: 20%;
    }
    .queue-table tr:nth-child(even) td {
        background: #f1f5f9;
    }
    .queue-table tr:hover td {
        background: #dbeafe;
    }
    .delete-btn {
        background: #ef4444;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 6px 10px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: background 0.3s ease, transform 0.1s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .delete-btn:hover {
        background: #dc2626;
        transform: scale(1.05);
    }
    .delete-btn:active {
        transform: scale(0.95);
    }
    .clear-queue-btn {
        background: #ef4444;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .clear-queue-btn:hover {
        background: #dc2626;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    @media (max-width: 768px) {
        .queue-table {
            display: block;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            width: 100%;
        }
        .queue-table th, .queue-table td {
            padding: 10px 12px;
            font-size: 0.9rem;
            min-width: 100px;
            min-height: 40px;
        }
        .delete-btn {
            padding: 5px 8px;
            font-size: 0.8rem;
        }
    }
    @media (max-width: 480px) {
        .queue-table thead {
            display: none;
        }
        .queue-table tr {
            display: block;
            margin-bottom: 0.75rem;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.5rem;
        }
        .queue-table td {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: none;
            padding: 8px;
            text-align: left;
            min-height: 36px;
            font-size: 0.85rem;
        }
        .queue-table td:before {
            content: attr(data-label);
            font-weight: 600;
            color: #1e40af;
            width: 40%;
            flex-shrink: 0;
        }
        .queue-table td:first-child {
            font-size: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }
        .delete-btn {
            width: 100%;
            justify-content: center;
            padding: 6px;
        }
    }
    .delete-btn:focus, .clear-queue-btn:focus {
        outline: 3px solid #93c5fd;
        outline-offset: 2px;
    }
    .stButton > button {
        border-radius: 8px;
        background: #2563eb;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #1e40af;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation tabs for switching between pages
page_order = ["📜 Instructions", "📋 Queue", "👁 Preview", "📊 Metrics", "📝 Logs"]
page = st.radio("nav", page_order, index=page_order.index(st.session_state.page), horizontal=True, label_visibility="collapsed")
st.session_state.page = page

# Instructions page with usage guide
if page.startswith("📜"):
    st.header("Instructions")
    st.markdown(
        """
        1. **Upload** a video (MP4, MOV, AVI, max 200MB) or a ZIP file containing JPEG/PNG images via the sidebar.
        2. **Select** one or more detectors and trackers.
        3. **Add to queue** — every detector-tracker pair becomes a row.
        4. **Run processing** to execute the first queued experiment and watch annotated frames in the *Preview* tab.
        5. Manage the queue in the *Queue* tab.

        **Supported Formats**:
        - Videos: MP4 (H.264), MOV, AVI (MJPEG). Requires FFmpeg (`bin/ffmpeg.exe`).
        - Images: ZIP file containing JPEG/PNG frames (no FFmpeg required).
        **Note**: Refreshing the browser resets the session, clearing the queue and results.
        """
    )
    if st.button("🔄 Reset session"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.session_state.logs = []
        st.session_state.logs.append(f"{datetime.now()}: Session reset")
        st.rerun()

# Queue page with full-width table
elif page.startswith("📋"):
    st.header("Experiment Queue")
    q = st.session_state.queue
    if not q:
        st.info("Queue is empty. Upload a video or image sequence and add detector-tracker pairs to start.")
    else:
        with st.container():
            # Start queue table
            st.markdown(
                """
                <table class='queue-table'>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Detector</th>
                            <th>Tracker</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                """,
                unsafe_allow_html=True,
            )
            # Iterate over queue items
            for i, job in enumerate(q[:]):  # Use a copy to avoid runtime modification issues
                st.markdown(
                    f"""
                    <tr>
                        <td data-label="Job #">{i+1}</td>
                        <td data-label="Detector">{job['detector']}</td>
                        <td data-label="Tracker">{job['tracker']}</td>
                        <td data-label="Action">
                            <button class='delete-btn' id='delete_{i}'>🗑️</button>
                        </td>
                    </tr>
                    """,
                    unsafe_allow_html=True,
                )
                # Handle delete button click
                if st.button("🗑️", key=f"delete_{i}", help=f"Delete job {i+1}"):
                    st.session_state.queue.pop(i)
                    st.session_state.logs.append(f"{datetime.now()}: Deleted job {i+1} from queue")
                    st.rerun()
            st.markdown("</tbody></table>", unsafe_allow_html=True)

        # Clear queue button
        if st.button("🗑️ Clear Queue", type="primary", key="clear_queue", help="Remove all jobs from the queue"):
            st.session_state.queue = []
            st.session_state.logs.append(f"{datetime.now()}: Cleared queue")
            st.rerun()

# Preview page for displaying processed frames
elif page.startswith("👁"):
    st.header("Preview")
    if st.session_state.preview_holder is None:
        st.info("Run processing to view frames with bounding boxes.")
    else:
        st.session_state.preview_holder.empty()
    if st.button("🛑 Stop Processing", use_container_width=True):
        st.session_state.stop_processing = True
        st.session_state.run = False
        st.session_state.preview_holder = None
        st.session_state.logs.append(f"{datetime.now()}: Stopped processing")
        st.rerun()

# Logs page for viewing and downloading session logs
elif page.startswith("📝"):
    st.header("Logs")
    if st.session_state.logs:
        st.text_area("Session Logs", "\n".join(st.session_state.logs), height=300)
        st.download_button(
            label="📥 Download Logs",
            data="\n".join(st.session_state.logs),
            file_name="session_logs.txt",
            mime="text/plain",
        )
    else:
        st.info("No logs available yet.")

# Metrics page for displaying job performance metrics
elif page.startswith("📊"):
    st.header("Metrics")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Jobs", st.session_state.job_index)
        with col2:
            st.metric("Total Queue Length", len(st.session_state.queue))

    if st.session_state.job_metrics:
        for job_key, metrics in st.session_state.job_metrics.items():
            job_idx, det_name, trk_name = job_key.split("_")
            job_idx = int(job_idx)
            st.subheader(f"Job {job_idx + 1}: {det_name} + {trk_name}")
            if not metrics["detections_per_frame"] or not metrics["tracks_per_frame"]:
                st.warning("No detections or tracks recorded. Check input quality or detector/tracker configuration.")
            st.session_state.show_plots = st.checkbox(
                f"Show Plots (Job {job_idx + 1})",
                value=st.session_state.show_plots,
                key=f"show_plots_{job_key}",
            )
            try:
                display_metrics(
                    detections_per_frame=metrics["detections_per_frame"],
                    tracks_per_frame=metrics["tracks_per_frame"],
                    fps_values=metrics["fps_values"],
                    start_idx=metrics.get("metrics_start_idx", 0),
                )
                metrics["metrics_start_idx"] = len(metrics["detections_per_frame"])
                st.session_state.logs.append(f"{datetime.now()}: Displayed metrics for Job {job_idx + 1}")
            except Exception as e:
                st.error(f"Failed to display metrics for Job {job_idx + 1}: {e}")
                st.session_state.logs.append(f"{datetime.now()}: Metrics display error for Job {job_idx + 1}: {e}")
            with st.container():
                if st.button(f"🔄 Update Metrics (Job {job_idx + 1})", key=f"update_metrics_{job_key}"):
                    with st.spinner(f"Updating metrics for Job {job_idx + 1}..."):
                        try:
                            display_metrics(
                                detections_per_frame=metrics["detections_per_frame"],
                                tracks_per_frame=metrics["tracks_per_frame"],
                                fps_values=metrics["fps_values"],
                                start_idx=metrics.get("metrics_start_idx", 0),
                            )
                            metrics["metrics_start_idx"] = len(metrics["detections_per_frame"])
                            st.session_state.logs.append(f"{datetime.now()}: Updated metrics for Job {job_idx + 1}")
                        except Exception as e:
                            st.error(f"Failed to update metrics for Job {job_idx + 1}: {e}")
                            st.session_state.logs.append(f"{datetime.now()}: Metrics update error for Job {job_idx + 1}: {e}")
            with st.container():
                st.subheader("Save Metrics")
                filename = st.text_input(
                    f"Metrics Filename (Job {job_idx + 1})",
                    value=f"metrics_job_{job_idx + 1}_{det_name}_{trk_name}.txt",
                    key=f"filename_{job_key}",
                )
                file_format = st.selectbox(
                    f"File Format (Job {job_idx + 1})",
                    ["txt", "json"],
                    key=f"format_{job_key}",
                )
                file_path = OUTPUT_DIR / cfg["output"]["metrics"] / filename
                if st.button(f"📥 Save Metrics (Job {job_idx + 1})", key=f"save_metrics_{job_key}"):
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        save_metrics_to_file(
                            metrics["detections_per_frame"],
                            metrics["tracks_per_frame"],
                            metrics["fps_values"],
                            str(file_path),
                            format=file_format,
                        )
                        st.success(f"Metrics saved to {file_path}")
                        st.session_state.logs.append(f"{datetime.now()}: Saved metrics for Job {job_idx + 1} to {file_path}")
                    except Exception as e:
                        st.error(f"Failed to save metrics for Job {job_idx + 1}: {e}")
                        st.session_state.logs.append(f"{datetime.now()}: Failed to save metrics for Job {job_idx + 1}: {e}")
    else:
        st.info("No metrics available yet. Run a job to view metrics.")

# Main processing engine
if st.session_state.run:
    error_container = st.empty()
    try:
        # Validate input and queue
        if not (st.session_state.video or st.session_state.image_sequence) or not st.session_state.queue:
            error_msg = "Upload a valid video or image sequence and add jobs to the queue first"
            error_container.error(error_msg)
            st.session_state.logs.append(f"{datetime.now()}: Run failed: {error_msg}")
            st.session_state.run = False
            st.rerun()
        if st.session_state.video and not ffmpeg_available:
            error_msg = f"FFmpeg is required for video processing. Place `ffmpeg.exe` in `{ffmpeg_path}` or use an image sequence."
            error_container.error(error_msg)
            st.session_state.logs.append(f"{datetime.now()}: Run failed: {error_msg}")
            st.session_state.run = False
            st.rerun()

        # Initialize video or image sequence processing
        fps = 20.0  # Default FPS for image sequences or fallback
        frame_count_total = 0
        width = 0
        height = 0
        if st.session_state.video:
            cap = cv2.VideoCapture(str(st.session_state.video_path))
            if not cap.isOpened():
                error_msg = f"Failed to open video file: {st.session_state.video_path}. The file may be corrupted or missing metadata."
                error_container.error(error_msg)
                st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
                st.session_state.run = False
                cap.release()
                st.rerun()
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            st.session_state.logs.append(f"{datetime.now()}: Video loaded - FPS: {fps}, Frames: {frame_count_total}")
        else:
            frame_count_total = len(st.session_state.image_sequence)
            if frame_count_total > 0:
                frame = cv2.imread(str(st.session_state.image_sequence[0]))
                if frame is not None:
                    height, width = frame.shape[:2]
                st.session_state.logs.append(f"{datetime.now()}: Image sequence loaded - Frames: {frame_count_total}")

        duration = frame_count_total / fps if fps > 0 else 12.0
        if duration > 60:
            st.session_state.preview_skip_frames = max(st.session_state.preview_skip_frames, int(fps / 2))

        total_jobs = len(st.session_state.queue)
        job_index = st.session_state.job_index

        if job_index >= total_jobs:
            error_msg = "No more jobs in queue"
            error_container.error(error_msg)
            st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
            st.session_state.run = False
            st.session_state.job_index = 0
            st.rerun()

        job = st.session_state.queue[job_index]
        det_name, trk_name = job["detector"], job["tracker"]
        st.session_state.logs.append(f"{datetime.now()}: Started job {job_index + 1}: {det_name} + {trk_name}")

        # Initialize preview holder
        if st.session_state.preview_holder is None:
            st.session_state.preview_holder = st.empty()

        # Display progress
        progress_bar = st.empty()
        progress_bar.progress(min(job_index / total_jobs, 1.0))

        # Display status
        status_panel = st.empty()
        status_panel.markdown(
            f"### 🔄 Processing queue item {job_index + 1} of {total_jobs}\n"
            f"- **Detector:** `{det_name}`\n"
            f"- **Tracker:** `{trk_name}`\n"
            f"- **📽️ Input:** `{st.session_state.video_path.name if st.session_state.video else 'Image Sequence'}`\n"
            f"- ⏳ Please wait while processing..."
        )

        # Initialize detector
        try:
            detector_map = {
                "YOLOv3": lambda: importlib.import_module("detectors.yolo").YOLOv3(
                    model=cfg["detectors"]["yolo"].get("model", "yolov3"),
                    conf_thresh=0.2,
                    nms_thresh=cfg["detectors"]["yolo"].get("nms_threshold", 0.45),
                ),
                "YOLOv5": lambda: importlib.import_module("detectors.yolov5").YOLOv5(
                    model_name=cfg["detectors"]["yolov5"].get("model", "yolov5s"),
                    conf_thresh=cfg["detectors"]["yolov5"].get("confidence_threshold", 0.4),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                "SSD": lambda: importlib.import_module("detectors.ssd").SSDDetector(
                    model=cfg["detectors"]["ssd"].get("model", "ssd300_vgg16"),
                    conf_thresh=cfg["detectors"]["ssd"].get("confidence_threshold", 0.5),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                "FasterRCNN": lambda: importlib.import_module("detectors.fasterrcnn").FasterRCNN(
                    model=cfg["detectors"]["fasterrcnn"].get("model", "fasterrcnn_resnet50_fpn"),
                    conf_thresh=cfg["detectors"]["fasterrcnn"].get("confidence_threshold", 0.5),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
            }
            detector = detector_map[det_name]()
            if det_name == "YOLOv5":
                detector.model.iou = cfg["detectors"]["yolov5"].get("nms_threshold", 0.4)
            st.session_state.logs.append(f"{datetime.now()}: Initialized detector: {det_name}")
        except Exception as e:
            error_msg = f"Detector initialization failed for Job {job_index + 1}: {e}"
            error_container.error(error_msg)
            st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
            st.session_state.job_index += 1
            if job_index + 1 < total_jobs:
                st.session_state.run = True
            else:
                st.session_state.run = False
                st.session_state.job_index = 0
            st.rerun()

        # Initialize tracker
        try:
            if trk_name == "SORT":
                from trackers.sort import SORTTracker as Tracker
                tracker = Tracker()
            elif trk_name == "DeepSORT":
                from trackers.deepsort import DeepSORTTracker as Tracker
                tracker = Tracker(
                    max_age=cfg["trackers"]["deepsort"].get("max_age", 30),
                    iou_threshold=cfg["trackers"]["deepsort"].get("iou_threshold", 0.3),
                )
            elif trk_name == "KCF":
                from trackers.kcf import KCFTracker as Tracker
                tracker = Tracker()
            elif trk_name == "MOSSE":
                from trackers.mosse import MOSSETracker as Tracker
                tracker = Tracker()
            elif trk_name == "MedianFlow":
                from trackers.medianflow import MedianFlowTracker as Tracker
                tracker = Tracker()
            else:
                raise ValueError(f"Unknown tracker: {trk_name}")
            st.session_state.logs.append(f"{datetime.now()}: Initialized tracker: {trk_name}")
        except Exception as e:
            error_msg = f"Tracker initialization failed for Job {job_index + 1}: {e}"
            error_container.error(error_msg)
            st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
            st.session_state.job_index += 1
            if job_index + 1 < total_jobs:
                st.session_state.run = True
            else:
                st.session_state.run = False
                st.session_state.job_index = 0
            st.rerun()

        # Initialize processing variables
        fps_values = []
        detections_per_frame = []
        tracks_per_frame = []
        frame_count = 0
        start_time = time.time()
        MAX_STORED_FRAMES = cfg["processing"].get("max_stored_frames", 1000)

        # Process video or image sequence
        out = None
        if st.session_state.video:
            try:
                output_path = OUTPUT_DIR / cfg["output"]["videos"] / f"output_job_{job_index + 1}_{det_name}_{trk_name}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if width > 0 and height > 0:
                    out = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (width, height)
                    )
                    if not out.isOpened():
                        st.session_state.logs.append(f"{datetime.now()}: Failed to initialize video writer: {output_path}")
                        out = None
                run_pair_iterator = run_pair(detector, tracker, st.session_state.video_path, save_output_path=output_path if out else None)
                for frm, detections, formatted_tracks in run_pair_iterator:
                    if st.session_state.stop_processing:
                        st.session_state.logs.append(f"{datetime.now()}: Interrupted job {job_index + 1}")
                        break
                    frame_count += 1
                    frame_start = time.time()
                    if out:
                        out.write(frm)
                    detections_per_frame.append(detections.tolist() if isinstance(detections, np.ndarray) else detections)
                    tracks_per_frame.append(formatted_tracks)
                    if len(detections_per_frame) > MAX_STORED_FRAMES:
                        detections_per_frame.pop(0)
                        tracks_per_frame.pop(0)
                    frame_time = time.time() - frame_start
                    fps = 1 / frame_time if frame_time > 0 else 0
                    if 0 < fps < 1000:
                        fps_values.append(fps)
                    if frame_count == 1 or frame_count % st.session_state.preview_skip_frames == 0:
                        try:
                            st.session_state.preview_holder.empty()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_DIR) as tmp:
                                cv2.imwrite(tmp.name, frm)
                                st.session_state.preview_holder.image(
                                    tmp.name,
                                    channels="BGR",
                                    caption=f"Frame {frame_count} of {frame_count_total} (Job {job_index + 1}: {det_name} + {trk_name})",
                                )
                            os.unlink(tmp.name)
                        except Exception as e:
                            error_container.error(f"Failed to display preview for frame {frame_count}: {e}")
                            st.session_state.logs.append(f"{datetime.now()}: Preview error for frame {frame_count}: {e}")
            except Exception as e:
                error_msg = f"Processing failed for Job {job_index + 1}: {e}"
                error_container.error(error_msg)
                st.session_state.logs.append(f"{datetime.now()}: Processing error: {e}")
                # Fallback to raw frame processing
                cap = cv2.VideoCapture(str(st.session_state.video_path))
                if cap.isOpened():
                    while cap.isOpened() and not st.session_state.stop_processing:
                        ret, frm = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        try:
                            detections = detector.detect(frm)
                            valid_detections = [
                                det for det in detections
                                if isinstance(det, (list, np.ndarray)) and len(det) == 5 and all(isinstance(v, (int, float)) for v in det)
                            ]
                            detections = np.array(valid_detections, dtype=np.float32) if valid_detections else np.array([], dtype=np.float32)
                            tracker_detections = detections[:, :4].tolist() if len(detections) > 0 else []
                            tracks = tracker.update(frm, tracker_detections)
                            formatted_tracks = [{"track_id": t["track_id"], "bbox": t["bbox"]} for t in tracks] if tracks else []
                            detections_per_frame.append(detections.tolist())
                            tracks_per_frame.append(formatted_tracks)
                            if len(detections_per_frame) > MAX_STORED_FRAMES:
                                detections_per_frame.pop(0)
                                tracks_per_frame.pop(0)
                            frm = draw_detections(frm, detections)
                            frm = draw_tracks(frm, formatted_tracks)
                            if frame_count == 1 or frame_count % st.session_state.preview_skip_frames == 0:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_DIR) as tmp:
                                    cv2.imwrite(tmp.name, frm)
                                    st.session_state.preview_holder.image(
                                        tmp.name,
                                        channels="BGR",
                                        caption=f"Raw Frame {frame_count} of {frame_count_total} (Fallback)",
                                    )
                                os.unlink(tmp.name)
                        except Exception as e:
                            st.session_state.logs.append(f"{datetime.now()}: Fallback processing error for frame {frame_count}: {e}")
                    cap.release()
                else:
                    error_msg = "Fallback failed: Cannot open video file"
                    error_container.error(error_msg)
                    st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
        else:
            try:
                for img_path in st.session_state.image_sequence:
                    if st.session_state.stop_processing:
                        st.session_state.logs.append(f"{datetime.now()}: Interrupted job {job_index + 1}")
                        break
                    frame_count += 1
                    frame_start = time.time()
                    frm = cv2.imread(str(img_path))
                    if frm is None:
                        st.session_state.logs.append(f"{datetime.now()}: Failed to load image {img_path}")
                        continue
                    detections = detector.detect(frm)
                    valid_detections = [
                        det for det in detections
                        if isinstance(det, (list, np.ndarray)) and len(det) == 5 and all(isinstance(v, (int, float)) for v in det)
                    ]
                    detections = np.array(valid_detections, dtype=np.float32) if valid_detections else np.array([], dtype=np.float32)
                    tracker_detections = detections[:, :4].tolist() if len(detections) > 0 else []
                    tracks = tracker.update(frm, tracker_detections)
                    formatted_tracks = [{"track_id": t["track_id"], "bbox": t["bbox"]} for t in tracks] if tracks else []
                    detections_per_frame.append(detections.tolist())
                    tracks_per_frame.append(formatted_tracks)
                    if len(detections_per_frame) > MAX_STORED_FRAMES:
                        detections_per_frame.pop(0)
                        tracks_per_frame.pop(0)
                    frm = draw_detections(frm, detections)
                    frm = draw_tracks(frm, formatted_tracks)
                    frame_time = time.time() - frame_start
                    fps = 1 / frame_time if frame_time > 0 else 0
                    if 0 < fps < 1000:
                        fps_values.append(fps)
                    if frame_count == 1 or frame_count % st.session_state.preview_skip_frames == 0:
                        try:
                            st.session_state.preview_holder.empty()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_DIR) as tmp:
                                cv2.imwrite(tmp.name, frm)
                                st.session_state.preview_holder.image(
                                    tmp.name,
                                    channels="BGR",
                                    caption=f"Frame {frame_count} of {frame_count_total} (Job {job_index + 1}: {det_name} + {trk_name})",
                                )
                            os.unlink(tmp.name)
                        except Exception as e:
                            error_container.error(f"Failed to display preview for frame {frame_count}: {e}")
                            st.session_state.logs.append(f"{datetime.now()}: Preview error for frame {frame_count}: {e}")
            except Exception as e:
                error_msg = f"Image sequence processing failed for Job {job_index + 1}: {e}"
                error_container.error(error_msg)
                st.session_state.logs.append(f"{datetime.now()}: Image sequence processing error: {e}")

        # Finalize job
        if out:
            out.release()
        end_time = time.time()
        elapsed = end_time - start_time
        est = (elapsed * (total_jobs - job_index - 1)) / (job_index + 1) if job_index + 1 < total_jobs else 0
        status_panel.markdown(
            f"### 🔄 Processing queue item {job_index + 1} of {total_jobs}\n"
            f"- **Detector:** `{det_name}`\n"
            f"- **Tracker:** `{trk_name}`\n"
            f"- **📽️ Input:** `{st.session_state.video_path.name if st.session_state.video else 'Image Sequence'}`\n"
            f"- ✅ Job completed\n"
            f"- **Estimated time to finish remaining jobs**: {round(est / 60, 2)} minutes"
        )

        # Store metrics
        job_key = f"{job_index}_{det_name}_{trk_name}"
        st.session_state.job_metrics[job_key] = {
            "detections_per_frame": detections_per_frame,
            "tracks_per_frame": tracks_per_frame,
            "fps_values": fps_values,
            "metrics_start_idx": 0,
            "detector": det_name,
            "tracker": trk_name,
        }

        # Display metrics
        if st.session_state.show_plots:
            try:
                display_metrics(detections_per_frame, tracks_per_frame, fps_values)
                st.session_state.logs.append(f"{datetime.now()}: Displayed metrics for Job {job_index + 1}")
            except Exception as e:
                error_msg = f"Failed to display metrics for Job {job_index + 1}: {e}"
                error_container.error(error_msg)
                st.session_state.logs.append(f"{datetime.now()}: Metrics display error: {e}")

        # Clear plot cache
        try:
            clear_plot_cache()
        except Exception as e:
            st.session_state.logs.append(f"{datetime.now()}: Failed to clear plot cache: {e}")

        # Log job completion
        st.success(f"✅ Job {job_index + 1} complete: {det_name} + {trk_name}")
        st.session_state.logs.append(f"{datetime.now()}: Finished job {job_index + 1}: {det_name} + {trk_name}")

        # Move to next job
        st.session_state.job_index += 1
        progress_bar.progress(min(st.session_state.job_index / total_jobs, 1.0))
        if job_index + 1 < total_jobs:
            st.session_state.run = True
        else:
            st.success("✅ All jobs completed")
            st.session_state.logs.append(f"{datetime.now()}: All queued jobs completed")
            st.session_state.run = False
            st.session_state.job_index = 0
        st.rerun()

    except Exception as e:
        error_msg = f"Unexpected error in processing: {e}"
        error_container.error(error_msg)
        st.session_state.logs.append(f"{datetime.now()}: {error_msg}")
        st.session_state.run = False
        st.rerun()

# Cleanup temporary directory on exit
def cleanup_temp_dir():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.logs.append(f"{datetime.now()}: Cleaned up temporary directory: {TEMP_DIR}")

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_temp_dir)