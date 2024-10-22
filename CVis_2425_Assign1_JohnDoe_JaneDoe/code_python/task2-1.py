import cv2
import numpy as np

# Load the image
image_path = '../data/clock_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest circular contour is the clock face
clock_contour = max(contours, key=cv2.contourArea)

# Create a mask for the clock face
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [clock_contour], -1, 255, -1)

# Extract the clock face using the mask
clock_face = cv2.bitwise_and(gray_image, gray_image, mask=mask)

# Find the center and radius of the clock face
(x, y), radius = cv2.minEnclosingCircle(clock_contour)
center = (int(x), int(y))
radius = int(radius)

# Draw the clock face contour and center on the original image
cv2.circle(image, center, radius, (0, 255, 0), 2)
cv2.circle(image, center, 5, (0, 0, 255), -1)

# Placeholder for clock hand detection and time extraction
# This part should include techniques to detect the clock hands and infer the time
# For now, let's assume we have detected the time as 10:10:30
detected_time = "10:10:30"

# Overlay the detected time on the original image
cv2.putText(image, detected_time, (center[0] - 50, center[1] + radius + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Save the result
output_path = '../data/clock_with_time.jpg'
cv2.imwrite(output_path, image)

# Display the result
cv2.imshow('Clock with Time', image)
cv2.waitKey(0)
cv2.destroyAllWindows()