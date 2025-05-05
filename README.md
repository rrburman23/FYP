# 🎓 Final Year Project – Object Detection & Tracking System

This project is my **Final Year Project** for a **BSc in Computer Science**. It features a real-time object detection and tracking system tailored for UAV (drone) footage. The system combines cutting-edge object detection models with multiple tracking algorithms, all wrapped in a clean, interactive **Streamlit** web app.

🔗 **Live Demo:** [Streamlit App](https://deployment-url.streamlit.app)

---

## 🔧 Features

- Real-time object detection and multi-object tracking
- Streamlit-based UI for uploading and processing UAV footage
- Support for YOLOv3, YOLOv5, SSD, FasterRCNN
- Multiple tracking algorithms (SORT, DeepSORT, KCF, MOSSE, MedianFlow)
- Visualisations and performance metrics

---

## 🗂️ Project Structure

FYP/
├── config/                # Application configuration files
│   └── default.yml        # Main YAML config for detectors, trackers, and processing
├── data/                  # Input and output data, like user uploaded videos
├── models/                # Pretrained models and weights (auto-downloaded)
│   ├── yolov3/            # YOLOv3 config, weights, and class names
│   └── ...                # Other models as needed
├── src/                   # Main source code
│   ├── __init__.py
│   ├── app.py             # Streamlit web app entry point
│   ├── main.py            # Terminal/CLI entry point
│   ├── detectors/         # Detector implementations (YOLOv3, YOLOv5, SSD, FasterRCNN)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── yolo.py
│   │   └── ...
│   ├── trackers/          # Tracker implementations (SORT, DeepSORT, KCF, MOSSE, MedianFlow)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sort.py
│   │   └── ...
│   ├── processing/        # Detection + tracking integration
│   │   ├── __init__.py
│   │   └── run_pair.py    # Runs the detector tracker pair on the uploaded video
│   ├── metrics/           # Evaluation and visualization modules
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── visualisation.py
│   └── util/              # Utility functions
│   │   ├── __init__.py
│       ├── paths.py
│       └── ...
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore rules for local files and temporary data

---

## 🚀 Getting Started (Local Setup)

Although the app is already deployed, you can also run it locally:

### 1. Clone the repository

```bash
git clone https://github.com/rrburman23/FYP.git
cd FYP
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run src/app.py
```

---

## 🤖 Supported Models

### Detectors

- YOLOv3 (automatically downloads weights and config)
- YOLOv5
- SSD
- FasterRCNN

### Trackers

- SORT
- DeepSORT
- MOSSE
- KCF
- MedianFlow

---

## 📊 Metrics & Visualisations

The app computes and displays:

- Total object detections
- Unique tracks and average duration
- Identity switches and fragmentation
- Average processing FPS
- Memory usage

Each processed video includes overlayed detection boxes and track IDs with colour-coded paths.

---

## 📌 Notes

Model files such as yolov3.weights, yolov3.cfg, and labels.names are automatically downloaded on first run.

This is a Final Year Project, completed as part of my undergraduate degree at Queen Mary University of London.

There are no plans for future development—this is the final version.
