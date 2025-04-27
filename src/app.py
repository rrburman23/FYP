"""
app.py
======

User workflow  ─────────────────────────────────────────────────────────
1. **Upload** a UAV video via the sidebar.                            ✓
2. **Tick** one or more detectors *and* trackers.                     ✓
3. Press **Add to queue** – every detector‑tracker pair becomes a row.✓
4. Hit **Run processing** to execute the first queued experiment      ✓
   and watch annotated frames stream in the *Preview* tab.            ✓
5. Manage the queue (delete rows / clear all) in the *Queue* tab.     ✓

Refreshing the browser always returns you to this Instructions tab and
wipes transient inputs.  
"""

from pathlib import Path
import streamlit as st
from utils.paths import DATA_DIR, load_config
from processing.run_pair import run_pair

# ───────────────────────────────────── Page setup ──────────────────────────
st.set_page_config(page_title="UAV Detection & Tracking",
                   page_icon="🚁",
                   layout="wide")
cfg = load_config()

# Available algorithms (only those implemented in detectors/ & trackers/)
DETECTORS = ["YOLOv3", "YOLOv5", "SSD", "FasterRCNN", "RetinaNet"]
TRACKERS  = ["DeepSORT", "SORT", "OpenCV", "ByteTrack"]

# ───────────────────────────── Session‑state scaffold ──────────────────────
_defaults = dict(
    page="📜 Instructions",    # current pseudo‑tab
    queue=[],                  # list[dict]
    run=False,                 # trigger flag
    video=None,                # UploadedFile object
    preview_holder=None        # st.empty() for frame streaming
)
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# ───────────────────────────────── Sidebar input ───────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Video uploader
    up = st.file_uploader("Upload Video (MP4 / MOV / AVI)",
                           type=("mp4", "mov", "avi"))
    if up:
        st.session_state.video = up

    # Algorithm selection
    det_sel = st.multiselect("Select Detectors", DETECTORS)
    trk_sel = st.multiselect("Select Trackers",  TRACKERS)

    # Add to queue
    if st.button("➕ Add to Queue", use_container_width=True):
        for d in det_sel:
            for t in trk_sel:
                job = {"detector": d, "tracker": t}
                if job not in st.session_state.queue:
                    st.session_state.queue.append(job)
        st.session_state.page = "📋 Queue"
        st.rerun()

    # Run button
    if st.button("▶️ Run Processing", use_container_width=True):
        st.session_state.run  = True
        st.session_state.page = "👁 Preview"
        st.rerun()

# ───────────────────────────────── CSS styling ─────────────────────────────
st.markdown(
"""
<style>
/* ── Radio → tab styling ─────────────────────────────────────────────── */
[data-testid="stRadio"]>div{display:flex;gap:.4rem;}
[data-testid="stRadio"] input{display:none;}          /* hide dots    */
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

/* ── Queue table ──────────────────────────────────────────────────────── */
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
unsafe_allow_html=True)

# ───────────────────────────────── Pseudo‑tabs ────────────────────────────
page_order = ["📜 Instructions", "📋 Queue", "👁 Preview", "📊 Metrics", "📝 Logs"]
page = st.radio("nav", page_order,
                index=page_order.index(st.session_state.page),
                horizontal=True, label_visibility="collapsed")
st.session_state.page = page   # persist current tab

# ───────────────────────────────── Instructions ───────────────────────────
if page.startswith("📜"):
    st.header("Instructions")
    st.markdown(__doc__.split("User workflow")[1])

    if st.button("🔄 Reset session"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.rerun()

# ───────────────────────────────── Queue tab ──────────────────────────────
elif page.startswith("📋"):
    st.header("Experiment Queue")

    q = st.session_state.queue
    if not q:
        st.info("Queue is empty.")
    else:
        # Build HTML table – single‑colour rows
        rows = "".join(
            f"<tr><td>{i+1}</td><td>{job['detector']}</td>"
            f"<td>{job['tracker']}</td>"
            f"<td><button class='trash' onclick=\"location.search='del={i}'\">🗑️</button></td></tr>"
            for i, job in enumerate(q)
        )
        st.markdown(
            "<table class='queue'><thead><tr><th>#</th><th>Detector</th>"
            "<th>Tracker</th><th></th></tr></thead><tbody>"
            + rows + "</tbody></table>",
            unsafe_allow_html=True,
        )

        # Clear‑all button
        if st.button("🗑️ Clear queue", type="primary"):
            st.session_state.queue = []
            st.rerun()

        # Handle per‑row delete via URL param
        if "del" in st.query_params:
            idx = int(st.query_params["del"])
            st.query_params.clear()
            if 0 <= idx < len(q):
                q.pop(idx)
            st.rerun()

# ───────────────────────────────── Preview tab ────────────────────────────
elif page.startswith("👁"):
    st.header("Preview")
    if st.session_state.preview_holder is None:
        st.info("Run processing to view frames.")

# ───────────────────────────────── Metrics / Logs placeholders ────────────
elif page.startswith("📊"):
    st.header("Metrics")
    st.info("Metrics will appear here once implemented.")
else:
    st.header("Logs")
    st.info("Logs placeholder – sent to console for now.")

# ─────────────────────────── Run processing engine ────────────────────────
if st.session_state.run:

    # Basic validation
    if st.session_state.video is None:
        st.sidebar.error("Please upload a video first.")
        st.session_state.run = False
        st.rerun()
    if not st.session_state.queue:
        st.sidebar.error("Queue is empty.")
        st.session_state.run = False
        st.rerun()

    # Persist uploaded video
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = DATA_DIR / st.session_state.video.name
    with open(video_path, "wb") as f:
        f.write(st.session_state.video.read())

    # Pop first job
    job = st.session_state.queue.pop(0)
    det_name, trk_name = job["detector"], job["tracker"]

    # ── Detector factory (SSD implemented, others stub for demo) ─────────
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
    elif det_name == "RetinaNet":
        from detectors.retinanet import RetinaNet as Detector
        detector = Detector()
    else:
        st.error(f"Detector **{det_name}** not available."); st.stop()

    # ── Tracker factory ────────────────────────────────────────────────
    if trk_name == "SORT":
        from trackers.sort import SORTTracker as Tracker
    elif trk_name == "DeepSORT":
        from trackers.deepsort import DeepSORTTracker as Tracker
    elif trk_name == "OpenCV":
        from trackers.opencv import OpenCVTracker as Tracker
    elif trk_name == "ByteTrack":
        from trackers.bytetrack import ByteTrackTracker as Tracker
    else:
        st.error(f"Tracker **{trk_name}** not available."); st.stop()
    tracker = Tracker()

    # Prepare preview holder
    holder = st.empty()
    st.session_state.preview_holder = holder

    # Stream frames with annotations
    for _, frm in run_pair(detector, tracker, video_path):
        holder.image(frm, channels="BGR")

    st.success("Processing complete ✅")
    st.session_state.run = False
    st.rerun()
