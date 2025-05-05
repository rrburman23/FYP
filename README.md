# 🎓 Object Detection & Tracking System

This **Final Year Project** for a **BSc in Computer Science** at Queen Mary University of London implements a real-time object detection and tracking system optimized for UAV (drone) footage. It integrates state-of-the-art detection models and tracking algorithms within an interactive **Streamlit** web application.

🔗 **Live Demo**: [Streamlit App](https://deployment-url.streamlit.app)

---

## ✨ Features

- Real-time object detection and multi-object tracking.
- Streamlit-based UI for uploading and processing UAV footage.
- Support for multiple detectors: YOLOv3, YOLOv5, SSD, FasterRCNN.
- Support for multiple trackers: SORT, DeepSORT, KCF, MOSSE, MedianFlow.
- Visualizations and performance metrics for processed videos.

---

## 📂 Project Structure

```text
FYP/
├── config/                   # Application configuration files
│   └── default.yml           # Main YAML config for detectors and trackers
├── data/                     # Input/output videos and metrics
├── models/                   # Pretrained models (auto-downloaded)
│   ├── yolov3/               # YOLOv3 configuration files
│   │   ├── yolov3.cfg        # YOLOv3 configuration
│   │   └── yolov3.weights    # YOLOv3 weights
│   └── ...                   # Other model directories
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── app.py                # Streamlit web interface
│   ├── main.py               # CLI entry point
│   ├── detectors/            # Detection implementations
│   │   ├── __init__.py       # Package initialization
│   │   ├── base.py           # Abstract detector interface
│   │   ├── yolo.py           # YOLOv3/YOLOv5 implementations
│   │   └── ...               # Other detectors
│   ├── trackers/             # Tracking implementations
│   │   ├── __init__.py       # Package initialization
│   │   ├── base.py           # Abstract tracker interface
│   │   ├── sort.py           # SORT implementation
│   │   └── ...               # Other trackers
│   ├── processing/           # Pipeline integration
│   │   ├── __init__.py       # Package initialization
│   │   └── run_pair.py       # Detector-tracker execution
│   ├── metrics/              # Performance evaluation
│   │   ├── __init__.py       # Package initialization
│   │   ├── evaluation.py     # Metric calculations
│    │   └── visualisation.py  # Plot generation
│   └── util/                 # Helper functions
│       ├── __init__.py       # Package initialization
│       ├── paths.py          # Path management
│       └── ...               # Other utilities
├── requirements.txt          # Python dependencies
└── .gitignore                # Ignore patterns for data, models, and caches
```

---

## 🚀 Getting Started (Local Setup)

Follow these steps to run the project locally.

### Prerequisites

- Python 3.8 or higher
- Git
- A virtual environment tool (e.g., `venv`)

### 1. Clone the Repository

```bash
git clone https://github.com/rrburman23/FYP.git
cd FYP
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run src/app.py
```

---

## 🤖 Supported Models

### Detectors

- **YOLOv3**: Automatically downloads weights and config.
- **YOLOv5**
- **SSD**
- **FasterRCNN**

### Trackers

- **SORT**
- **DeepSORT**
- **MOSSE**
- **KCF**
- **MedianFlow**

---

## 📊 Metrics & Visualizations

The application computes and displays the following metrics:

- Total object detections.
- Unique tracks and average track duration.
- Identity switches and fragmentation.
- Average processing FPS.
- Memory usage.

Processed videos include overlaid detection boxes and track IDs with color-coded paths.

---

## 📝 Notes

- Model files (e.g., `yolov3.weights`, `yolov3.cfg`, `labels.names`) are automatically downloaded on the first run.
- This project was completed as part of an undergraduate degree and is not planned for further development.
- For any issues, refer to the GitHub repository.
