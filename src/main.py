"""
Command-line interface for batch processing videos with detector-tracker pairs.

Features:
- Real-time progress bar
- Metrics calculation
- Output video with annotations
- Metrics export to JSON/CSV

Usage:
    python src/main.py --det YOLOv5 --trk SORT --video data/test_12s_360p.mp4 
                       --output_video output.mp4 --metrics_file metrics.json
"""

import argparse
import cv2
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

from util.paths import load_config
from processing.run_pair import run_pair
from metrics.evaluation import calculate_tracking_metrics, track_fps_and_memory
from util.visualise import draw_detections, draw_tracks 

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def import_class(path: str):
    """Dynamic import from 'module.submodule.ClassName' string"""
    module_path, class_name = path.rsplit(".", 1)
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)

def main():
    parser = argparse.ArgumentParser(description="CLI for detection & tracking")
    parser.add_argument("--det", required=True, choices=["YOLOv3", "YOLOv5", "SSD", "FasterRCNN"],
                       help="Detector to use")
    parser.add_argument("--trk", required=True, choices=["SORT", "DeepSORT", "KCF", "MOSSE", "MedianFlow"],
                       help="Tracker to use")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output_video", help="Output video path (MP4)")
    parser.add_argument("--metrics_file", help="Output metrics file (JSON/CSV)")
    parser.add_argument("--show", action="store_true", help="Show real-time preview (requires GUI)")

    args = parser.parse_args()
    start_time = time.time()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"{Colors.FAIL}Error: Could not open video {args.video}{Colors.ENDC}")
        return

    # Load configuration
    cfg = load_config()

    # Load detector and tracker
    try:
        Detector = import_class(f"detectors.{args.det.lower()}.{args.det}")
        conf_thresh = cfg["detectors"][args.det.lower()]["confidence_threshold"]
        detector = Detector(conf_thresh=conf_thresh)

        Tracker = import_class(f"trackers.{args.trk.lower()}.{args.trk}Tracker")
        tracker = Tracker()
    except Exception as e:
        print(f"{Colors.FAIL}Initialization failed: {str(e)}{Colors.ENDC}")
        return

    # Setup video writer
    writer = None
    if args.output_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Init metrics
    metrics = {
        'detections_per_frame': [],
        'tracks_per_frame': [],
        'fps_values': [],
        'start_time': datetime.now().isoformat(),
        'video': args.video,
        'detector': args.det,
        'tracker': args.trk
    }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(frame, detections)

            metrics['detections_per_frame'].append([d.tolist() for d in detections])
            metrics['tracks_per_frame'].append([t.tolist() for t in tracks])
            metrics['fps_values'].append(1 / (time.time() - frame_start + 1e-9))

            if writer or args.show:
                annotated = frame.copy()
                annotated = draw_detections(annotated, detections)
                annotated = draw_tracks(annotated, tracks)

                if writer:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow('Preview', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"{Colors.WARNING}Preview interrupted by user{Colors.ENDC}")
                        break

            pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Processing interrupted by user{Colors.ENDC}")
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        pbar.close()

    # Final metrics
    total_time = time.time() - start_time
    avg_fps, memory_usage = track_fps_and_memory(metrics['fps_values'])

    metrics.update({
        'processing_time': total_time,
        'avg_fps': avg_fps,
        'max_memory_mb': memory_usage,
        'end_time': datetime.now().isoformat()
    })

    det_metrics = calculate_tracking_metrics(
        metrics['detections_per_frame'],
        metrics['tracks_per_frame']
    )

    metrics.update({
        'total_detections': det_metrics[0],
        'unique_tracks': det_metrics[1],
        'avg_track_length': det_metrics[2],
        'id_switches': det_metrics[3],
        'fragmentation': det_metrics[4]
    })

    # Save to file
    if args.metrics_file:
        with open(args.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"{Colors.OKGREEN}Metrics saved to {args.metrics_file}{Colors.ENDC}")

    # Print summary
    print(f"\n{Colors.OKCYAN}―――――――――――――――― Results ――――――――――――――――{Colors.ENDC}")
    print(f"{Colors.BOLD}Video:{Colors.ENDC} {args.video}")
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC} {args.det} + {args.trk}")
    print(f"\n{Colors.BOLD}Performance:{Colors.ENDC}")
    print(f"  • Avg FPS: {avg_fps:.1f}")
    print(f"  • Max Memory: {memory_usage:.1f} MB")
    print(f"  • Processing Time: {total_time:.1f}s")
    
    print(f"\n{Colors.BOLD}Tracking Metrics:{Colors.ENDC}")
    print(f"  • Detections: {metrics['total_detections']}")
    print(f"  • Unique Tracks: {metrics['unique_tracks']}")
    print(f"  • Avg Track Length: {metrics['avg_track_length']:.1f} frames")
    print(f"  • ID Switches: {metrics['id_switches']}")
    print(f"  • Fragmentation: {metrics['fragmentation']}")

if __name__ == "__main__":
    main()
