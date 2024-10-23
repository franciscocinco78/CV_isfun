import cv2
import numpy as np

SHOW_MIDDLE_STEPS = False

def getAngleFromCenter(line, _center):
    x1, y1, x2, y2 = line[0][0]
    dist1 = np.sqrt((x1 - _center[0])**2 + (y1 - _center[1])**2)
    dist2 = np.sqrt((x2 - _center[0])**2 + (y2 - _center[1])**2)
    if dist1 > dist2:
        farthest_point = (x1, y1)
    else:
        farthest_point = (x2, y2)
    angle = np.degrees(np.arctan2(farthest_point[1] - _center[1], farthest_point[0] - _center[0]))
    return angle

def showImage(d,i):
    cv2.imshow(d, i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'data/clock.png'
image = cv2.imread(image_path)

# using HSV color space to isolate the clock
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color in HSV
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
# Create a mask to isolate green regions
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
# Invert the mask to isolate the clock face
clock_face_mask = cv2.bitwise_not(green_mask)
# Apply the mask to the original image
clock_face_isolated = cv2.bitwise_and(image, image, mask=clock_face_mask)
# Update the image variable to the isolated clock face
image = clock_face_isolated
if SHOW_MIDDLE_STEPS:
    showImage('Clock face with background removed',image)

# Convert the isolated clock face to grayscale, will be later used to detect min and hour pointers
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Convert the grayscale image to pure black and white
_, black_and_white_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(black_and_white_image, 70, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
clock_contour = max(contours, key=cv2.contourArea)
# Find the center and radius of the clock face
(x, y), radius = cv2.minEnclosingCircle(clock_contour)
center = (int(x), int(y))
radius = int(radius)
# print('Radius:', int(radius))
if SHOW_MIDDLE_STEPS:
    cv2.circle(image, center, radius, (0, 0, 255), 2)
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    showImage('Clock face with center and outer contour highlighted in red',image)

###### -> Get seconds pointer by isolating reds
# Define the range for red color in HSV
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
# Create masks to isolate red regions
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)
# Apply the mask to the original image
red_isolated = cv2.bitwise_and(image, image, mask=red_mask)
if SHOW_MIDDLE_STEPS:
    showImage('HSV Red component Isolated', red_isolated)

edges = cv2.Canny(red_isolated, 70, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
# print(lines)
# print(getAngleFromCenter(lines,center))
seconds = int((( getAngleFromCenter(lines,center) +90) % 360) / 6)

if SHOW_MIDDLE_STEPS:
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(red_isolated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(lines)
    showImage('Line detected on isolated red component, shown in green.',red_isolated)

# Convert the red isolated image to grayscale and threshold the gray_red_isolated image 
# to get a binary mask
gray_red_isolated = cv2.cvtColor(red_isolated, cv2.COLOR_BGR2GRAY)
_, red_mask_binary = cv2.threshold(gray_red_isolated, 1, 255, cv2.THRESH_BINARY)
# Overlay the red isolated content as white pixels on the black and white image
black_and_white_image[red_mask_binary == 255] = 255
# This image will be used to easily identify min and hour pointers
if SHOW_MIDDLE_STEPS:
    showImage('Black and white with seconds pointer removed.',black_and_white_image)
    
edges = cv2.Canny(black_and_white_image, 70, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)
closest_lines = []
angles = []
i=0
dof=3 # degrees of freedom, lines with angles within dof degrees of each other are considered the same
for line in lines:
    x1, y1, x2, y2 = line[0]
    # h = sqrt( c^2 + c1^2)
    d1=np.sqrt( (x1-center[0])**2 + (y1-center[1])**2 )
    d2=np.sqrt( (x2-center[0])**2 + (y2-center[1])**2 )
    if (np.sqrt( (x1-x2)**2 + (y1-y2)**2 )) > 183:
        continue
    ang=0
    # a line ahs 2 points
    if d1 < radius/2:   # point 1 is closer to the center
        x2+= -1*(x1-center[0]) # move the line to the center, 
        y2+= -1*(y1-center[1]) # just so angles are consistent between iterations
        ang= np.degrees(np.arctan2(y2 - center[1], x2 - center[0])) + 90
    elif d2 < radius/2: # point 2 is closer to the center
        # if x1 < center[0]:    # Crop lines part behind clock center coordinates, to prevent same size lines from being detected
        #     line[0][0] = center[0]
        # if y1 < center[1]:    # pointer's quadrant should be taken into account
        #     line[0][1] = center[1]
        x1+= -1*(x2-center[0]) 
        y1+= -1*(y2-center[1])
        ang= np.degrees(np.arctan2(y1 - center[1], x1 - center[0])) + 90
    if len(angles) > 0:
        if angles[0]+dof > ang and angles[0]-dof < ang:
            continue
        elif len(angles) > 1:
            if angles[1]+dof > ang and angles[1]-dof < ang:
                continue
            elif ang != 0:
                print(f"Unexpected third angle!!! {ang}")
                continue
        elif ang != 0:
            angles.append(ang)
            closest_lines.append(line)
            # print(line)
    elif ang != 0:
        angles.append(ang)
        closest_lines.append(line)
        # print(line)
    i+=1

minutes = 0
hours = 0
x1, y1, x2, y2 = closest_lines[0][0]
d0=(np.sqrt( (x1-x2)**2 + (y1-y2)**2 ))
x1, y1, x2, y2 = closest_lines[1][0]
d1=(np.sqrt( (x1-x2)**2 + (y1-y2)**2 ))
if SHOW_MIDDLE_STEPS:
    print(f"Line 1: {closest_lines[0]}")
    print(f"Line 1 length: {d0:.2f}")
    print(f"Line 2: {closest_lines[1]}")
    print(f"Line 2 length: {d1:.2f}")
if d0 > d1:
    minutes = int((angles[0] % 360) / 6)
    hours = int((angles[1] % 360) / 30)
else:
    minutes = int((angles[1] % 360) / 6)
    hours = int((angles[0] % 360) / 30)
if SHOW_MIDDLE_STEPS:
    print(f"Time: {hours}:{minutes}:{seconds}")

if SHOW_MIDDLE_STEPS:
    image_copy = image.copy()
    if closest_lines is not None:
        for line in closest_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    showImage('Detected Closest Lines on Clock Face', image_copy)

detected_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
# Overlay the detected time on the original image
image = cv2.imread(image_path)
cv2.putText(image, detected_time, (center[0] - 50, center[1] + radius + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Save the result
output_path = '../data/task2-2_clock_with_time.jpg'
cv2.imwrite(output_path, image)

# Display the result
cv2.imshow('Clock with Time', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

