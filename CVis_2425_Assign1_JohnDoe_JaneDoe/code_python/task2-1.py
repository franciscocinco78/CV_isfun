import cv2
import numpy as np

image_path = 'data/clock.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# GaussianBlur to reduce noise and improve contour detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#use Canny edge detection to find edges in the image
edges = cv2.Canny(blurred_image, 70, 150)
# cv2.imshow('Blurred Image', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#largest circular contour is the clock face
clock_contour = max(contours, key=cv2.contourArea)
# create a mask for the clock face
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [clock_contour], -1, 255, -1)
# Extract the clock face using the mask
clock_face = cv2.bitwise_and(gray_image, gray_image, mask=mask)
# cv2.imshow('Clock Face',clock_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
# Find the center and radius of the clock face
(x, y), radius = cv2.minEnclosingCircle(clock_contour)
center = (int(x), int(y))
radius = int(radius)

# Draw the clock face contour and center on the original image
cv2.circle(image, center, radius, (0, 255, 0), 2)
cv2.circle(image, center, 5, (0, 0, 255), -1)

#############################################

# Display the clock face
# cv2.imshow('Clock Face', clock_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# using Hough Line Transform to detect lines in the clock face
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Function to calculate distance from a point to a line
def distance_from_center(line, center):
    x1, y1, x2, y2 = line[0]
    return abs((y2 - y1) * center[0] - (x2 - x1) * center[1] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

# Find the three lines closest to the center
lines = sorted(lines, key=lambda line: distance_from_center(line, center))[:5]
# Print the length of each line
lengths=np.zeros(5)
j=0  
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    lengths[j]=length
    print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has length {length:.2f}")
    j+=1
    
# Calculate the angle from the center to the farthest point of each line
angles = np.zeros(5)  
j=0
for i, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
    dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
    if dist1 > dist2:
        farthest_point = (x1, y1)
    else:
        farthest_point = (x2, y2)
    angle = np.degrees(np.arctan2(farthest_point[1] - center[1], farthest_point[0] - center[0]))
    angles[j]=angle
    print(f"Line {i+1} farthest point angle: {angle:.2f}")
    j+=1
    
# Draw the three closest lines on the original image with different colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
for i, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    # Calculate the distance from the center to the farthest edge of the line
    dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
    dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
    farthest_distance = max(dist1, dist2)
    print(f"Line {i+1} farthest distance from center: {farthest_distance:.2f}", end=", ")
    total_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"Line {i+1} total length: {total_length:.2f}")
    # Draw the line on the image
    image_copy = image.copy()
    cv2.line(image_copy, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
    # Display the image with the current line
    # cv2.imshow(f'Clock with Line {i+1}', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#############################################
# Create three variables to store the sum of lengths of lines with close angle values
sum_length_1 = 0
sum_length_2 = 0
sum_length_3 = 0
ang1 = angles[0]
ang2 = 0
ang3 = 0
i=0
for ang in angles:
    if ang1-15 <= ang <= ang1+15:
        sum_length_1 += lengths[i]
        # print(f"ang1: {ang1}, ang: {ang}, length: {lengths[i]}, sum_length_1: {sum_length_1}")
    else:
        if ang2==0:
            ang2 = ang
        if ang2-15 <= ang <= ang2+15:
            sum_length_2 += lengths[i]
            # print(f"ang2: {ang2}, ang: {ang}, length: {lengths[i]}, sum_length_1: {sum_length_2}")
        elif ang3==0:
            ang3 = ang
        if ang3-15 <= ang <= ang3+15:
            sum_length_3 += lengths[i]
            # print(f"ang3: {ang3}, ang: {ang}, length: {lengths[i]}, sum_length_1: {sum_length_3}") 
        # else:
        #     print(f"ang: {ang}, length: {lengths[i]}")
    i=i+1


hour=0
min=0
sec=0
def getType(val1, val2, val3, ii):
    if i==1:
        val1=0
    elif ii==2:
        val2=0
    elif ii==3:
        val3=0
    if val1 > val2 and val1 > val3:
        return int(((ang1+90) % 360) / 6), 1
    elif val2 > val1 and val2 > val3:
        return int(((ang2+90) % 360) / 6), 2
    elif val3 > val1 and val3 > val2:
        return int(((ang3+90) % 360) / 6), 3
    else:
        print("Unexpected Error")
        return 0, 0

sec, pos = getType(sum_length_1, sum_length_2, sum_length_3,0)
min, i = getType(0, sum_length_2, sum_length_3,pos)
if (pos==1 and i==2) or (pos==2 and i ==1):
    hour = int(((ang3+90) % 360) / 30)
elif (pos==1 and i==3) or (pos==3 and i ==1):
    hour = int(((ang2+90) % 360) / 30)
elif (pos==2 and i==3) or (pos==3 and i ==2):
    hour = int(((ang1+90) % 360) / 30)

print(f"Hour: {hour}, Minute: {min}, Second: {sec}")

detected_time = f"{hour:02d}:{min:02d}:{sec:02d}"

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