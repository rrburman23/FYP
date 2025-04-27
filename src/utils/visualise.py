import cv2
from typing import List, Tuple

COLOURS = [(255,0,0),(0,255,0),(0,0,255)]

def draw_tracks(frame, tracks: List[Tuple[int,int,int,int,int]]):
    for idx, (tid, x1, y1, x2, y2) in enumerate(tracks):
        colour = COLOURS[idx % len(COLOURS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f'ID:{tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return frame
