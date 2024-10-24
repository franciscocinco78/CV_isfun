import cv2
import numpy as np
import math

# Callback function for the trackbars (required but doesn't need to do anything)
def nothing(x):
    pass

# Function to detect clock face and hands
def detect_clock(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)



    # Detect the clock face using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100)

    if circles is not None:
        # Convert detected circles to integers
        circles = np.round(circles[0, :]).astype("int")
        return img, blur, circles
    else:
        return img, blur, None

# Create a window for the sliders and display
cv2.namedWindow('Edges')

# Create trackbars for Canny thresholds
cv2.createTrackbar('Canny Thresh 1', 'Edges', 0, 600, nothing)
cv2.createTrackbar('Canny Thresh 2', 'Edges', 0, 600, nothing)
cv2.createTrackbar('Canny Thresh 3', 'Edges', 0, 20, nothing)

# Image path
image_path = 'E:/CV/pratica/project/CV_isfun/CVis_2425_Assign1_JohnDoe_JaneDoe/data/clock.png'
img, blur, circles = detect_clock(image_path)

while True:
    # Get current positions of the trackbars
    # thresh1 = cv2.getTrackbarPos('Canny Thresh 1', 'Edges')
    # thresh2 = cv2.getTrackbarPos('Canny Thresh 2', 'Edges')
    # thresh3 = cv2.getTrackbarPos('Canny Thresh 3', 'Edges')
    counter1, counter2, counter3 = 0, 0, 0

    if circles is not None:
        for (x, y, r) in circles:
            # Draw the clock face circle
            cv2.circle(img, (x, y), r, (0, 0, 0), 2)

            # Mask the clock face to focus on the hands
            mask = np.zeros_like(blur)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Apply the mask to the blurred image
            masked_img = cv2.bitwise_and(blur, mask)
            kernel1 = 2*(np.ones((5, 5), np.float32))/30
            masked_img1 = cv2.filter2D(masked_img, -1, kernel1)

            # Detect edges in the clock face region (Canny Edge Detection)
            edges = cv2.Canny(masked_img1, 170 , 210, apertureSize=3)

            # Detect lines (clock hands) using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=60, maxLineGap=1)

            if lines is not None:

                xt_s, yt_s, xt_h, yt_h, xt_m, yt_m = 0, 0, 0, 0, 0, 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    size = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    print(line, size)
                    if size < 98:
                        #cv2.line(img, (x1, y1), (x, y), (0, 0, 255), 2)
                        counter1 = counter1 + 1
                        xt_s = xt_s + x1
                        yt_s = yt_s + y1

                    elif size < 99:
                        #cv2.line(img, (x1, y1), (x, y), (255, 0, 0), 2)
                        counter2 = counter2 + 1
                        xt_h = xt_h + x1
                        yt_h = yt_h + y1
                    else:
                        #cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
                        counter3 = counter3 + 1
                        xt_m = xt_m + x2
                        yt_m = yt_m + y2

    xt_s, yt_s = int(xt_s / counter1), int(yt_s / counter1)
    xt_h, yt_h = int(xt_h / counter2), int(yt_h / counter2)
    xt_m, yt_m = int(xt_m / counter3), int(yt_m / counter3)
    cv2.line(img, (xt_s, yt_s), (x, y), (0, 0, 255), 2)  # Red for second hand
    cv2.line(img, (xt_h, yt_h), (x, y), (255, 0, 0), 2)  # Blue for hour hand
    cv2.line(img, (x, y), (xt_m, yt_m), (0, 255, 0), 2)  # Green for minute hand
    print(f"Counter1: {counter1}, Counter2: {counter2}, Counter3: {counter3}")
    second_angle = np.arctan2(y - yt_s, x - xt_s) * 180 / np.pi -90
    hour_angle = np.arctan2(y - yt_h, x - xt_h) * 180 / np.pi -90
    minute_angle = np.arctan2(yt_m - y, xt_m- x) * 180 / np.pi +90
    hour = int(((hour_angle + 360) % 360) / 360 * 12) if hour_angle is not None else 0
    minute = int(((minute_angle + 360) % 360) / 360 * 60) if minute_angle is not None else 0
    second = int(((second_angle + 360) % 360) / 360 * 60) if second_angle is not None else 0
    print(f"Detected time: {hour:02}:{minute:02}:{second:02}")
    # Display the edges and detected hands
    cv2.imshow('Edges', edges)
    cv2.imshow('Detected Clock Hands', img)


    # Break the loop on pressing ''
    if cv2.waitKey(0):
        break


    # Clean up windows
    cv2.destroyAllWindows()
