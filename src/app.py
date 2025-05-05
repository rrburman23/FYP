# app.py

"""
Streamlit GUI for UAV Object Detection & Tracking

User Workflow:
    1. Upload a UAV video from the sidebar
    2. Select one or more object detectors and trackers
    3. Add combinations to the experiment queue
    4. Run processing — annotated frames stream in the *Preview* tab
    5. Manage the experiment queue via the *Queue* tab
    6. View metrics and logs

Notes:
- Refreshing the browser resets session state.
- Designed for modular extensibility (plug-and-play detectors/trackers).
"""

# ────────────────────────────── Imports ─────────────────────────────────

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import time
import torch
torch.classes.__path__ = []
import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from utils import DATA_DIR, load_config, draw_detections, draw_tracks
from processing.run_pair import run_pair
from metrics import display_metrics


# ────────────────────────────── Imports ─────────────────────────────────
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
torch.classes.__path__ = []

# ─────────────────────── Streamlit Configuration ───────────────────────
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🚁",
    layout="wide"
)

# ───────────────────────────── Configuration ───────────────────────────
cfg = load_config()  # Load YAML or JSON config

# Supported detector and tracker classes
DETECTORS = ["YOLOv3", "YOLOv5", "SSD", "FasterRCNN"]
TRACKERS = ["DeepSORT", "SORT", "KCF", "MOSSE", "MedianFlow"]

# ────────────────────── Session State Initialization ───────────────────
_defaults = dict(
    page="📜 Instructions",
    queue=[],
    run=False,
    video=None,
    preview_holder=None,
    stop_processing=False,
    job_index=0,
)

for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# ───────────────────────────── Sidebar UI ──────────────────────────────
with st.sidebar:
    st.header("Configuration")

    up = st.file_uploader("Upload Video (MP4 / MOV / AVI)", type=("mp4", "mov", "avi"))
    if up:
        st.session_state.video = up

    det_sel = st.multiselect("Select Detectors", DETECTORS)
    trk_sel = st.multiselect("Select Trackers", TRACKERS)

    if st.button("➕ Add to Queue", use_container_width=True):
        for d in det_sel:
            for t in trk_sel:
                job = {"detector": d, "tracker": t}
                if job not in st.session_state.queue:
                    st.session_state.queue.append(job)
        st.session_state.page = "📋 Queue"
        st.rerun()

    if st.button("▶️ Run Processing", use_container_width=True):
        st.session_state.run = True
        st.session_state.page = "👁 Preview"
        st.rerun()

# ─────────────────────────── UI Styling (CSS) ──────────────────────────
st.markdown(
    """
    <style>
    /* Tab styling */
    [data-testid="stRadio"]>div{display:flex;gap:.4rem;}
    [data-testid="stRadio"] input{display:none;}          
    [data-testid="stRadio"] label{margin:0;}
    [data-testid="stRadio"] div[role='radiogroup']>div{
      padding:.55rem 1.2rem; cursor:pointer;
      border-bottom:2px solid transparent; transition:all .15s ease;
    }
    [data-testid="stRadio"] div[role='radiogroup']>div:hover{
      background:#f5f5f5;}
    [data-testid="stRadio"] input:checked+div{
      color:#0a84ff; font-weight:600; border-color:#0a84ff;}
    
    table.queue{border-collapse:collapse;width:100%;}
    table.queue th,table.queue td{
      border:1px solid #d0d0d0; padding:8px 14px; text-align:center;}
    table.queue th{
      background:#0a84ff; color:#fff; font-weight:600;}
    button.trash{
      font-size:18px; color:#666; background:none; border:none; cursor:pointer;
      transition:all .15s ease;}
    button.trash:hover{color:#d11; font-size:22px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Navigation Tabs ───────────────────────────
page_order = ["📜 Instructions", "📋 Queue", "👁 Preview", "📊 Metrics", "📝 Logs"]
page = st.radio("nav", page_order, index=page_order.index(st.session_state.page), horizontal=True, label_visibility="collapsed")
st.session_state.page = page

# ────────────────────────────── Instructions ────────────────────────────
if page.startswith("📜"):
    st.header("Instructions")
    st.markdown("""
1. **Upload** a UAV video via the sidebar.  
2. **Select** one or more detectors **and** trackers.  
3. Press **Add to queue** — every detector-tracker pair becomes a row.  
4. Hit **Run processing** to execute the first queued experiment  
   and watch annotated frames stream in the *Preview* tab.  
5. Manage the queue in the *Queue* tab.

**Note**: Refreshing the browser resets session state.
    """)

    if st.button("🔄 Reset session"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.rerun()

# ──────────────────────────────── Queue ────────────────────────────────
elif page.startswith("📋"):
    st.header("Experiment Queue")
    q = st.session_state.queue
    if not q:
        st.info("Queue is empty.")
    else:
        rows = "".join(
            f"<tr><td>{i+1}</td><td>{job['detector']}</td><td>{job['tracker']}</td><td><button class='trash' onclick=\"location.search='del={i}'\">🗑️</button></td></tr>"
            for i, job in enumerate(q)
        )
        st.markdown(
            "<table class='queue'><thead><tr><th>#</th><th>Detector</th><th>Tracker</th><th></th></tr></thead><tbody>" + rows + "</tbody></table>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear queue", type="primary"):
            st.session_state.queue = []
            st.rerun()

# ─────────────────────────────── Preview ───────────────────────────────
elif page.startswith("👁"):
    st.header("Preview")
    if st.session_state.preview_holder is None:
        st.info("Run processing to view frames.")
    if st.button("🛑 Stop Processing", use_container_width=True):
        st.session_state.stop_processing = True

# ─────────────────────────────── Logs ──────────────────────────────────
elif page.startswith("📝"):
    st.header("Logs")
    if "logs" in st.session_state:
        st.text_area("Session Logs", "\n".join(st.session_state.logs), height=300)
    else:
        st.info("No logs available yet.")

# ─────────────────────────────── Metrics ───────────────────────────────
elif page.startswith("📊"):
    st.header("Metrics")
    st.metric("Processed Jobs", st.session_state.job_index)
    st.metric("Total Queue Length", len(st.session_state.queue))

# ────────────────────────────── Run Engine ─────────────────────────────
if st.session_state.run:

    if st.session_state.video is None or not st.session_state.queue:
        st.sidebar.error("Upload a video and add jobs to the queue first.")
        st.session_state.run = False
        st.rerun()

    # Save uploaded video to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = DATA_DIR / st.session_state.video.name
    with open(video_path, "wb") as f:
        f.write(st.session_state.video.read())

    total_jobs = len(st.session_state.queue)
    job_index = st.session_state.job_index

    if job_index < total_jobs:
        job = st.session_state.queue[job_index]
        det_name, trk_name = job["detector"], job["tracker"]

        st.session_state.setdefault("logs", []).append(f"▶️ Started job {job_index + 1}: {det_name} + {trk_name}")

        if st.session_state.preview_holder is None:
            st.session_state.preview_holder = st.empty()

        progress_bar = st.progress(job_index / total_jobs)
        status_panel = st.empty()
        status_panel.markdown(
            f"### 🔄 Processing Item {job_index + 1} of {total_jobs}\n"
            f"- **Detector:** `{det_name}`\n"
            f"- **Tracker:** `{trk_name}`\n"
            f"- **📽️ Video:** `{video_path.name}`\n"
            f"- ⏳ Please wait while processing..."
        )

        # Detector factory
        try:
            detector_map = {
                "SSD": lambda: __import__('detectors.ssd').ssd.SSDDetector(conf_thresh=cfg["processing"]["confidence_threshold"]),
                "YOLOv3": lambda: __import__('detectors.yolo').yolo.YOLOv3(),
                "YOLOv5": lambda: __import__('detectors.yolov5').yolov5.YOLOv5(),
                "FasterRCNN": lambda: __import__('detectors.fasterrcnn').fasterrcnn.FasterRCNN(),
            }
            detector = detector_map[det_name]()
        except Exception as e:
            st.error(f"❌ Detector init failed: {e}")
            st.session_state.logs.append(f"❌ Detector error: {e}")
            st.stop()

        # Tracker factory
        try:
            if trk_name == "SORT":
                from trackers.sort import SORTTracker as Tracker
                tracker = Tracker()
            elif trk_name == "DeepSORT":
                from trackers.deepsort import DeepSORTTracker as Tracker
                tracker = Tracker()
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
        except Exception as e:
            st.error(f"❌ Tracker init failed: {e}")
            st.session_state.logs.append(f"❌ Tracker error: {e}")
            st.stop()

        # Run processing loop
        start_time = time.time()
        for frm, detections, tracks in run_pair(detector, tracker, video_path):
            if st.session_state.stop_processing:
                st.session_state.logs.append(f"⏸️ Interrupted job {job_index + 1}")
                st.session_state.run = False
                st.session_state.stop_processing = False
                st.success("Processing interrupted ⏸️")
                break
            
            progress_bar.progress((job_index + 1) / total_jobs)
            st.session_state.preview_holder.image(frm, channels="BGR", caption=f"Processing job {job_index + 1}")

        # Completion status
        end_time = time.time()
        elapsed = end_time - start_time
        est = (elapsed * (total_jobs - job_index - 1)) / (job_index + 1)
        st.session_state.preview_holder.markdown(f"**Estimated time to finish**: {round(est / 60, 2)} minutes")

        st.success(f"✅ Job {job_index + 1} complete")
        st.session_state.logs.append(f"✅ Finished job {job_index + 1}: {det_name} + {trk_name}")

        if job_index + 1 < total_jobs:
            st.session_state.job_index += 1
            st.session_state.run = True
            st.rerun()
        else:
            st.success("✅ All jobs completed")
            st.session_state.logs.append("✅ All queued jobs completed")
            st.session_state.run = False
            st.session_state.job_index = 0

# ─────────────────────── Additional Handling ───────────────────────
# Handle corrupted video uploads
def check_video_validity(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    return True

# Save processed video with annotations
def save_processed_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
