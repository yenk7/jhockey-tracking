#!/usr/bin/env python3
"""Generate a printable 9x6 checkerboard pattern for camera calibration."""
import cv2
import numpy as np

rows, cols = 7, 10  # 7×10 squares = 6×9 inner corners
square_px = 100

img = np.zeros((rows * square_px, cols * square_px), dtype=np.uint8)
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 0:
            img[r*square_px:(r+1)*square_px, c*square_px:(c+1)*square_px] = 255

# Add white border + label
bordered = np.ones((img.shape[0] + 100, img.shape[1] + 100), dtype=np.uint8) * 255
bordered[50:50+img.shape[0], 50:50+img.shape[1]] = img
cv2.putText(bordered, "9x6 inner corners (10x7 squares) - print on letter/A4",
            (50, bordered.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

cv2.imwrite("checkerboard_9x6.png", bordered)
print(f"Saved checkerboard_9x6.png ({bordered.shape[1]}x{bordered.shape[0]} px)")
print("Print on letter/A4 paper, tape to a flat rigid surface (cardboard/clipboard).")
