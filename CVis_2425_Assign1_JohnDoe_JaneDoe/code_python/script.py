import cv2
import numpy as np
import math
# Function to detect clock face and hands, then plot the hands
image_path = 'E:\CV\pratica\project\CV_isfun\CVis_2425_Assign1_JohnDoe_JaneDoe\data\clock.png'
    # Read the image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect the clock face using HoughCircles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2,100)

if circles is not None:
    # Convert detected circles to integers
    circles = np.round(circles[0, :]).astype("int")
    
    for (x, y, r) in circles:
        # Draw the clock face circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

        # Mask the clock face to focus on the hands
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Apply the mask to the blurred image
        masked_img = cv2.bitwise_and(gray, gray, mask=mask)

        # Detect edges in the clock face region (Canny Edge Detection)
        edges = cv2.Canny(masked_img, 50, 150)

        # Detect lines (clock hands) using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=1)

        # if lines is not None:

        #     for line in lines:
        #         print("Number of detected lines:", len(lines))
        #         print("Lines:", line)
        #         x1, y1, x2, y2 = line[0]
        #         print(f"Line length: {np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)}")
        #         line_size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
lines[2:6,0]=0,0,0,0
lines[7:,0]=0,0,0,0
x1, y1, x2, y2 = lines[0,0]
print("Number of detected lines:", len(lines))
print("Lines:", lines[6,0])
print(f"Line length: {np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)}")
line_size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
cv2.line(img, (x1, y1), (x, y), (0, 0, 255), 2)
second_angle = math.degrees(math.atan2(y - y1, x - x1)) -90
x1, y1, x2, y2 = lines[1,0]
hour_angle = math.degrees(math.atan2(y - y1, x - x1)) -90
cv2.line(img, (x1, y1), (x, y), (0, 255, 0), 2)
x1, y1, x2, y2 = lines[6,0]
minute_angle = math.degrees(math.atan2(y2 - y, x2 - x)) +90
hour = int(((hour_angle + 360) % 360) / 360 * 12) if hour_angle is not None else 0
minute = int(((minute_angle + 360) % 360) / 360 * 60) if minute_angle is not None else 0
second = int(((second_angle + 360) % 360) / 360 * 60) if second_angle is not None else 0

cv2.line(img, (x, y), (x2, y2), (255, 0, 0), 2)
print(f"Detected time: {hour:02}:{minute:02}:{second:02}")
cv2.imshow("Detected Clock Hands", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
