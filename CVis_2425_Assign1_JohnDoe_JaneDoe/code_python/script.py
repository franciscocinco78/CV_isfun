import cv2
import numpy as np
import math

img = cv2.imread('data/clock.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw the lines (clock hands)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Calculate the angle of the line relative to the center of the clock
        angle = math.degrees(math.atan2(y2 - y, x2 - x))
        print(f"Hand angle: {angle}")

cv2.imshow('Clock', edges)

cv2.waitKey(0)

cv2.destroyAllWindows()