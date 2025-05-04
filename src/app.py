"""
app.py

Main runner for Streamlit GUI.

User Workflow:
    1. Upload a UAV video via the sidebar.  
    2. Select one or more detectors and trackers.  
    3. Press Add to queue — every detector-tracker pair becomes a row.  
    4. Hit Run processing to execute the first queued experiment  
    and watch annotated frames stream in the *Preview* tab.  
    5. Manage the queue (delete rows / clear all) in the *Queue* tab.  

    Refreshing the browser always returns you to this Instructions tab and
    wipes transient inputs.
"""

import time
import torch
import streamlit as st
from pathlib import Path
import cv2
from utils import DATA_DIR, load_config, draw_detections, draw_tracks
from processing.run_pair import run_pair


# ───────────────────────────────── Page setup ──────────────────────────
st.set_page_config(page_title="Object Detection & Tracking", page_icon="🚁", layout="wide")  # Changed to camera icon

cfg = load_config()  # Load configuration settings from external file

# Available algorithms (only those implemented in detectors/ & trackers/)
DETECTORS = ["YOLOv3", "YOLOv5", "SSD", "FasterRCNN"]
TRACKERS = ["DeepSORT", "SORT", "GOTURN", "MedianFlow"]

# ───────────────────────────── Session‑state scaffold ──────────────────────
# Initialize session state variables for application flow and persistence
_defaults = dict(
    page="📜 Instructions",  # current pseudo‑tab
    queue=[],  # list of queued detector-tracker jobs
    run=False,  # trigger flag for running experiments
    video=None,  # UploadedFile object for video
    preview_holder=None,  # st.empty() for frame streaming
    stop_processing=False,  # Flag to stop processing prematurely
    job_index=0,
)

# Set default session state values if not already set
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# ───────────────────────────────── Sidebar input ───────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Video uploader
    up = st.file_uploader("Upload Video (MP4 / MOV / AVI)", type=("mp4", "mov", "avi"))
    if up:
        st.session_state.video = up

    # Algorithm selection for detectors and trackers
    det_sel = st.multiselect("Select Detectors", DETECTORS)
    trk_sel = st.multiselect("Select Trackers", TRACKERS)

    # Add to queue button to append selected detector-tracker pairs to the queue
    if st.button("➕ Add to Queue", use_container_width=True):
        for d in det_sel:
            for t in trk_sel:
                job = {"detector": d, "tracker": t}
                if job not in st.session_state.queue:
                    st.session_state.queue.append(job)
        
        # DEBUG: print amount of items added to queue and the current position in the queue
        print(f"Queue updated: {len(st.session_state.queue)} items in queue.")
        print(f"Current position in queue: {len(st.session_state.queue)}")

        st.session_state.page = "📋 Queue"
        st.rerun()

    # Run button to start processing the queue
    if st.button("▶️ Run Processing", use_container_width=True):
        st.session_state.run = True
        st.session_state.page = "👁 Preview"
        st.rerun()

# ───────────────────────────────── CSS styling ─────────────────────────────
# CSS for styling of UI components like tabs, buttons, and queues
st.markdown(
    """
<style>
/* Styling for tabs and queues */
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
[data-testid="stRadio"] input:checked+div::after{
  content:""; position:absolute; left:0; bottom:-2px;
  width:100%; height:2px; background:#0a84ff; animation:slide .25s;}
@keyframes slide{from{width:0}to{width:100%}}

/* Queue Table */
table.queue{border-collapse:collapse;width:100%;}
table.queue th,table.queue td{
  border:1px solid #d0d0d0; padding:8px 14px; text-align:center;}
table.queue th{
  background:#0a84ff; color:#fff; font-weight:600;}
button.trash{
  font-size:18px; color:#666; background:none; border:none; cursor:pointer;
  transition:all .15s ease;}
button.trash:hover{
  color:#d11; font-size:22px;}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────── Pseudo‑tabs ────────────────────────────
# Navigation tabs to switch between pages
page_order = ["📜 Instructions", "📋 Queue", "👁 Preview", "📊 Metrics", "📝 Logs"]
page = st.radio(
    "nav",
    page_order,
    index=page_order.index(st.session_state.page),
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.page = page  # persist current tab

# Instructions page
if page.startswith("📜"):
    st.header("Instructions")
    st.markdown("""
1. **Upload** a UAV video via the sidebar.  
2. **Select** one or more detectors **and** trackers.  
3. Press **Add to queue** — every detector-tracker pair becomes a row.  
4. Hit **Run processing** to execute the first queued experiment  
   and watch annotated frames stream in the *Preview* tab.  
5. Manage the queue (delete rows / clear all) in the *Queue* tab.  

Refreshing the browser always returns you to this Instructions tab and
wipes transient inputs.
""")

    # Reset button for clearing session state
    if st.button("🔄 Reset session"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.rerun()

# Queue tab
elif page.startswith("📋"):
    st.header("Experiment Queue")

    q = st.session_state.queue
    if not q:
        st.info("Queue is empty.")
    else:
        # Display queue as a table
        rows = "".join(
            f"<tr><td>{i+1}</td><td>{job['detector']}</td>"
            f"<td>{job['tracker']}</td>"
            f"<td><button class='trash' onclick=\"location.search='del={i}'\">🗑️</button></td></tr>"
            for i, job in enumerate(q)
        )
        st.markdown(
            "<table class='queue'><thead><tr><th>#</th><th>Detector</th>"
            "<th>Tracker</th><th></th></tr></thead><tbody>" + rows + "</tbody></table>",
            unsafe_allow_html=True,
        )

        # Clear-all button
        if st.button("🗑️ Clear queue", type="primary"):
            st.session_state.queue = []
            st.rerun()

# Preview tab
elif page.startswith("👁"):
    st.header("Preview")
    if st.session_state.preview_holder is None:
        st.info("Run processing to view frames.")

    # Add Stop/Interrupt button for processing
    if st.button("🛑 Stop Processing", use_container_width=True):
        st.session_state.stop_processing = True

# Logs tab
elif page.startswith("📝 Logs"):
    st.header("Logs")
    if "logs" in st.session_state:
        st.text_area("Session Logs", "\n".join(st.session_state.logs), height=300)
    else:
        st.info("No logs available yet.")

# Metrics tab
elif page.startswith("📊 Metrics"):
    st.header("Metrics")
    st.metric("Processed Jobs", st.session_state.job_index)
    st.metric("Total Queue Length", len(st.session_state.queue))
    # Add more custom metrics as needed

    

# ───────────────────────────────── Run processing engine ────────────────────────
if st.session_state.run:

    # Basic validation for video and queue
    if st.session_state.video is None:
        st.sidebar.error("Please upload a video first.")
        st.session_state.run = False
        st.rerun()
    if not st.session_state.queue:
        st.sidebar.error("Queue is empty.")
        st.session_state.run = False
        st.rerun()

    # Persist uploaded video to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = DATA_DIR / st.session_state.video.name
    with open(video_path, "wb") as f:
        f.write(st.session_state.video.read())
    print(f"[run_pair] Video file saved at: {video_path}")

    # Retrieve the current job from the queue
    total_jobs = len(st.session_state.queue)
    job_index = st.session_state.get("job_index", 0)

    if job_index < total_jobs:
        job = st.session_state.queue[job_index]
        det_name, trk_name = job["detector"], job["tracker"]

        # Logging Added
        st.session_state.setdefault("logs", [])
        st.session_state.logs.append(f"▶️ Started job {job_index + 1}: {det_name} + {trk_name}")

        # Initialize preview frame container if not set
        if st.session_state.preview_holder is None:
            st.session_state.preview_holder = st.empty()

        # Progress and status display elements
        progress_bar = st.progress(job_index / total_jobs)
        status_panel = st.empty()

        # Live job metadata display for the user
        status_panel.markdown(
            f"### 🔄 Processing Item {job_index + 1} of {total_jobs}\n"
            f"- **Detector:** `{det_name}`\n"
            f"- **Tracker:** `{trk_name}`\n"
            f"- **📽️ Video:** `{video_path.name}`\n"
            f"- ##⏳ Please wait while processing..."
        )

        # ─── Detector factory ─────────────────────────────────────────────
        try:
            if det_name == "SSD":
                from detectors.ssd import SSDDetector as Detector
                detector = Detector(conf_thresh=cfg["processing"]["confidence_threshold"])
            elif det_name == "YOLOv3":
                from detectors.yolo import YOLOv3 as Detector
                detector = Detector()
            elif det_name == "YOLOv5":
                from detectors.yolov5 import YOLOv5 as Detector
                detector = Detector()
            elif det_name == "FasterRCNN":
                from detectors.fasterrcnn import FasterRCNN as Detector
                detector = Detector()
            else:
                st.error(f"Detector **{det_name}** not available.")
                st.session_state.logs.append(f"❌ Unknown detector specified: {det_name}")
                st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize detector '{det_name}': {e}")
            st.session_state.logs.append(f"❌ Detector init failed for job {job_index + 1} ({det_name}): {e}")
            st.stop()

        # ─── Tracker factory ──────────────────────────────────────────────
        try:
            if trk_name == "SORT":
                from trackers.sort import SORTTracker as Tracker
                tracker = Tracker()
            elif trk_name == "DeepSORT":
                from trackers.deepsort import DeepSORTTracker as Tracker
                tracker = Tracker()
            elif trk_name == "GOTURN":
                from trackers.goturn import GOTURNTracker as Tracker
                model_path = "path_to_goturn_model"  # Replace with actual model path
                tracker = Tracker(model_path)
            elif trk_name == "MedianFlow":
                tracker = cv2.TrackerMedianFlow_create()
            else:
                st.error(f"Tracker **{trk_name}** not available.")
                st.session_state.logs.append(f"❌ Unknown tracker specified: {trk_name}")
                st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize tracker '{trk_name}': {e}")
            st.session_state.logs.append(f"❌ Tracker init failed for job {job_index + 1} ({trk_name}): {e}")
            st.stop()

        # ─── Run Processing ───────────────────────────────────────────────
        start_time = time.time()

       # ─── Frame streaming loop — interruption-safe ────────────────────────
        for frm, detections, tracks in run_pair(detector, tracker, video_path):
            if st.session_state.stop_processing:
                st.session_state.logs.append(f"⏸️ Job {job_index + 1} interrupted by user")
                st.session_state.run = False
                st.session_state.stop_processing = False
                st.success("Processing interrupted ⏸️")
                break

            # Annotate frame with detections and tracks
            frm = draw_detections(frm, detections)  # Draw detection bounding boxes
            frm = draw_tracks(frm, tracks)          # Draw tracking bounding boxes and IDs

            # Update progress bar
            progress_bar.progress((job_index + 1) / total_jobs)

            # Show the current annotated frame in the preview panel
            st.session_state.preview_holder.image(frm, channels="BGR", caption=f"Processing frame {job_index + 1}")


        # ─── Completion and Time Estimation ───────────────────────────────
        end_time = time.time()
        elapsed_time = end_time - start_time
        estimated_time = (elapsed_time * (total_jobs - job_index - 1)) / (job_index + 1)

        st.session_state.preview_holder.markdown(
            f"**Estimated time to finish**: {round(estimated_time / 60, 2)} minutes",
            unsafe_allow_html=True,
        )

        # ─── Job Completion Logging ───────────────────────────────────────
        st.success(f"Processing complete for Queue Item {job_index + 1} ✅")
        st.session_state.logs.append(f"✅ Finished job {job_index + 1}: {det_name} + {trk_name}")

        # ─── Move to Next Job or Wrap Up ──────────────────────────────────
        if job_index + 1 < total_jobs:
            st.session_state.job_index += 1
            st.session_state.run = True  # Continue processing next item
            st.rerun()
        else:
            st.success("✅ All queued jobs completed.")
            st.session_state.logs.append("✅ All queued jobs completed.")
            st.session_state.run = False
            st.session_state.job_index = 0
