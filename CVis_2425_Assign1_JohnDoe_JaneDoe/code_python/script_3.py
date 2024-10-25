import cv2
import numpy as np
import os
import math

# Paths to images and threshold files
image_path = 'project/CV_isfun/CVis_2425_Assign1_JohnDoe_JaneDoe/data/clock.png'
threshold_file_second = 'threshold_values_seconds.txt'
threshold_file_minute_hour = 'threshold_values_minute_hour.txt'

# Function to read HSV threshold values from a file
def load_threshold_values(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            values = [int(line.strip()) for line in file.readlines()]
            return tuple(values[:6])
    else:
        # Default values if file doesn't exist
        return (0, 50, 50, 10, 255, 255)

# Function to save HSV threshold values to a file
def save_threshold_values(file_path, values):
    with open(file_path, 'w') as file:
        for value in values:
            file.write(f"{value}\n")



# Function to determine the hour from the angles
def determine_time(hour_angle, minute_angle, second_angle):
    # Convert angles to approximate times
    hours = int((hour_angle % 360) / 30)  # 360 degrees / 12 hours = 30 degrees/hour
    minutes = int((minute_angle % 360) / 6)  # 360 degrees / 60 minutes = 6 degrees/minute
    seconds = int((second_angle % 360) / 6)  # 360 degrees / 60 seconds = 6 degrees/second

    return hours, minutes, seconds

# Load initial threshold values
(lower_h, lower_s, lower_v, upper_h, upper_s, upper_v) = load_threshold_values(threshold_file_minute_hour)

# Set up trackbars to adjust HSV ranges
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H", "Trackbars", lower_h, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", lower_s, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", lower_v, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", upper_h, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", upper_s, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", upper_v, 255, nothing)

# Load and process the image
image = cv2.imread(image_path)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100, minRadius=100)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        center = (x, y)  # Clock's center
        cv2.circle(image, center, r, (0, 0, 0), 2)

        mask = np.zeros_like(blurred)
        cv2.circle(mask, center, r, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(blurred, mask)

        KERNEL = np.array([[0, 0, 0], [0, 1.5, 0], [0, 0, 0]])
        masked_img_1 = cv2.filter2D(masked_img, -1, KERNEL)
        hsv = cv2.cvtColor(masked_img_1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

        # Get current HSV range from trackbars
        lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
        lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
        lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
        upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
        upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
        upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

        lower_red = np.array([104, 17, 90])
        upper_red = np.array([231, 255, 255])
        lower_2 = np.array([0, 0, 0])
        upper_2 = np.array([179, 255, 158])

        mask1 = cv2.inRange(hsv, lower_red, upper_red)  # For second hand
        mask2 = cv2.inRange(hsv2, lower_2, upper_2)     # For minute and hour hands
        kernel1 = np.ones((3, 3), np.uint8)
        mask2 = cv2.erode(mask2, kernel1, iterations=3)
        mask1_1 = cv2.dilate(mask1, kernel1, iterations=1)
        mask3 = mask2 - mask1_1

        mask3 = cv2.erode(mask3, kernel1, iterations=4)
        mask3 = cv2.dilate(mask3, kernel1, iterations=3)

        # Create a mask to black out the area outside the circle
        height, width = mask3.shape[:2]
        circle_mask = np.zeros((height, width), dtype=np.uint8)

        # Fill the circle area with white (255)
        cv2.circle(circle_mask, center, r, (255), thickness=-1)

        # Apply the mask to keep the area inside the circle and black out the outside
        mask3 = cv2.bitwise_and(mask3, mask3, mask=circle_mask)

        # Find contours
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Second hand
        contours2, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Minute and hour hands

        # Get the largest contour as the second hand
        second_tip = None
        if contours:
            largest_second_contour = max(contours, key=cv2.contourArea)
            second_tip = tuple(largest_second_contour[largest_second_contour[:, :, 1].argmax()][0])  # Bottom-most point

        # Get two largest contours for hour and minute hands
        if len(contours2) >= 2:
            contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
            minute_hand_contour = contours2[0]  # Largest contour is minute hand
            hour_hand_contour = contours2[1]    # Second largest is hour hand

            minute_tip = tuple(minute_hand_contour[minute_hand_contour[:, :, 1].argmin()][0])  # Top-most point for minute
            hour_tip = tuple(hour_hand_contour[hour_hand_contour[:, :, 1].argmin()][0])        # Top-most point for hour

            # Calculate angles for minute, hour, and second hands
            if second_tip:
                print(f"Second Hand: {second_tip}")
                print(f"center: {center}")
                # cv2.circle(image, second_tip, 5, (0, 0, 255), -1)
                # cv2.circle(image, center, 5, (0, 0, 255), -1)   
                second_angle = math.degrees(math.atan2(y - second_tip[1], x - second_tip[0])) -90
                print(f"Second Hand Angle: {second_angle:.2f} degrees")
                second = int(((second_angle + 360) % 360) / 360 * 60) if second_angle is not None else 0
                print(f"Second Hand: {second}")

            if minute_tip:
                print(f"Minute Hand: {minute_tip}")
                print(f"center: {center}")
                # cv2.circle(image, minute_tip, 5, (255, 0, 0), -1)
                # cv2.circle(image, center, 5, (255, 0, 0), -1)
                minute_angle = math.degrees(math.atan2(y - minute_tip[1], x - minute_tip[0])) -90
                print(f"Minute Hand Angle: {minute_angle:.2f} degrees")
                minute = int(((minute_angle + 360) % 360) / 360 * 60) if minute_angle is not None else 0
                print(f"Minute Hand: {minute}")


            if hour_tip:
                print(f"Hour Hand: {hour_tip}")
                print(f"center: {center}")
                # cv2.circle(image, hour_tip, 5, (0, 0, 255), -1)
                # cv2.circle(image, center, 5, (0, 0, 255), -1)
                hour_angle = math.degrees(math.atan2(y - hour_tip[1], x - hour_tip[0])) -90
                print(f"Hour Hand Angle: {hour_angle:.2f} degrees")
                hour = int(((hour_angle + 360) % 360) / 360 * 12) if hour_angle is not None else 0
                print(f"Hour Hand: {hour}")


            # Plot contours for debugging
            # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)      # Second hand in green
            # cv2.drawContours(image, [minute_hand_contour], -1, (255, 0, 0), 2)  # Minute hand in blue
            # cv2.drawContours(image, [hour_hand_contour], -1, (0, 0, 255), 2)    # Hour hand in red

cv2.putText(image, f"Time: {hour:02d}:{minute:02d}:{second:02d}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


cv2.imshow('Masked Image', masked_img_1)
cv2.imshow('HSV Image', hsv)
cv2.imshow('Mask1', mask1)   # Second hand
cv2.imshow('Mask3', mask3)   # Minute and hour hands
cv2.imshow('Input Image', image)



# Save threshold values and clean up
save_threshold_values(threshold_file_minute_hour, [lower_h, lower_s, lower_v, upper_h, upper_s, upper_v])
cv2.waitKey(0)
cv2.destroyAllWindows()
