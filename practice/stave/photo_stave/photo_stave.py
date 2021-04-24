import cv2
import numpy as np
from pathlib import Path
DATA_PATH = 'data'

img = cv2.imread(str(Path(DATA_PATH, 'normal.jpg')))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

min_line_len = 300
max_line_gap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                        minLineLength=min_line_len, maxLineGap=max_line_gap)

for x1, y1, x2, y2 in lines[:, 0, :]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imwrite(str(Path(DATA_PATH, 'normal_green.jpg')), img)
