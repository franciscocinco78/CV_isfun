import cv2
import numpy as np
import csv
import os

# Set to True to show intermediate steps of the process, and show extra information
SHOW_MIDDLE_STEPS = False

# center of the clock face, will not be calculated each frame, as it is assumed to be constant
center = None 
radius = 0  # same situation as 'center'
last_angle_sec = 400
last_angle_min = 400
last_angle_hour = 400
last_hour = 60
last_min = 60
last_sec = 60
error_count = 0
last_arc = None

# useful fuction to log relevant data when a critial error occurs
def logData(img):
    global center, radius
    print(f"center: {center}")
    print(f"radius: {radius}")
    cv2.imwrite(f'troublesome_frame.png', img)

# If current time is consistent save the global vars.
# If not, increment the error count
# @todo implement this functionallity
def store_last_angle_hour_min(angles0, angles1, hh, mm, ss):
    pass

def is_hm_angle_expected(ang):
    global last_angle_min, last_angle_hour
    if last_angle_min==400 and last_angle_hour==400:
        return True
    mm=last_angle_min
    hh=last_angle_hour
    if hh +15 > 360:
        hh=0
    if mm +15 > 360:
        mm=0
    if ((ang < hh + 15 and ang > hh - 10) or 
        (ang < mm + 15 and ang > mm - 10)):
            return True # expected angle
    else:
        return False
        
# Function to get the angle of a line with respect to the center of the clock face
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

# bw_img: binary image
# cc: center of the clock
# small_radius: radius of the small circle (cropped area)
# large_radius: radius of the large circle (cropped area)
def get_arc(bw_img, cc, small_radius, large_radius):
    # Create a mask with the larger circle
    mm = cv2.circle(np.zeros_like(bw_img), cc, large_radius, 255, -1) # .
    # Subtract the smaller circle from the mask
    mm = cv2.circle(mm, cc, small_radius+5, 0, -1) # adjust value added to small_radius
    cropped = cv2.bitwise_and(bw_img, bw_img, mask=mm)
    # Draw a full white circle with the radius of the small circle
    white_circle = np.zeros_like(bw_img) # full black image
    cv2.circle(white_circle, cc, small_radius, 255, -1) # draw white circle
    outer_mm = cv2.circle(np.ones_like(bw_img) * 255, cc, large_radius-2, 0, -1) # adjust value subtracted to large_radius
    # Apply the mask to the cropped binary image
    return cv2.bitwise_or(cropped, cv2.bitwise_or(outer_mm, white_circle))

# cc_img1: & cc_img2: binary images of the cropped arc
# cc_img1: binary image of the previous frame
# cc_img2: binary image of the current frame
# orig_img: original image
# cc: center of the clock
def get_current_minute_pointer_angle(cc_img1, cc_img2, orig_img, cc):
    difference = cv2.absdiff(cc_img1, cc_img2)
    SHOW_ALL = SHOW_MIDDLE_STEPS
    if SHOW_ALL:
        showImage('get minutes -> cc_img1', cc_img1)
        showImage('get minutes -> cc_img2', cc_img2)
        showImage('get minutes -> Difference', difference)
    # Remove single pixels without neighbors
    kernel = np.ones((3, 3), np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
    if SHOW_ALL:
        showImage('get minutes -> Difference clean', difference)

    # Find contours of the objects in the difference image
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    highlighted_image = orig_img.copy()
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # Red contours with thickness 2
    if SHOW_ALL:
        showImage('get minutes -> highlighted_image', highlighted_image)
    # Find and print the center of mass of each object
    angles = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            ang = np.degrees(np.arctan2(cY - cc[1], cX - cc[0]))+90
            angles.append(ang)
            # print(f"Center of mass: ({cX}, {cY}), angle: {ang:.1f}")
            cv2.circle(highlighted_image, (cX, cY), 5, (255, 0, 0), -1)  # Draw
    if SHOW_ALL:
        showImage('get minutes -> highlighted_image', highlighted_image)
    if len(angles) > 1:
        if max(angles) > 170 and min(angles) < 10: # pointer crossed 00
            a = []
            for i in angles:
                if i < 11:
                    a.append(i)
            if len(a) == 1:
                return a[0]
            else:
                angles_copy = a.copy()
                angles_copy.pop(angles_copy.index(max(a)))
                closest_angle = min(angles_copy, key=lambda x: abs(x - max(a)))
                return (closest_angle + max(a)) / 2
        
        angles_copy = angles.copy()
        angles_copy.pop(angles_copy.index(max(angles)))
        closest_angle = min(angles_copy, key=lambda x: abs(x - max(angles)))
        if closest_angle > (max(angles)-4):   # se os angulos forem proximos
            return (closest_angle + max(angles)) / 2 # devolver a media
        else:
            return max(angles)
    else:
        return angles[0]

def showImage(d,i):
    cv2.imshow(d, i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process the image and detect the time
# Most of the code is the same as in script of task 2-2 
def doStuff(image):
    global center, radius # allow access to global variables
    global last_angle_min, last_angle_hour
    
    image_input = image.copy() # this copy is used to log the frame in case of error
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
    # Convert the grayscale image to pure black and white, ignore returned threshold value
    _, black_and_white_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    
    if center is None: # center not yet defined? Find it!
        # It will remain constant for all remaining frames, saving processing time and preventing unthinkable bugs that made me create this
        edges = cv2.Canny(black_and_white_image, 70, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        clock_contour = max(contours, key=cv2.contourArea)
        # Find the center and radius of the clock face
        (x, y), radius = cv2.minEnclosingCircle(clock_contour)
        center = (int(x), int(y))
        radius = int(radius)
    
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        for j in range(90,49,-10): # No lines detected?
            # Strategy 2, make all pixels solid red and reduce the threshold to detect lines
            red_isolated[np.where((red_isolated != [0, 0, 0]).all(axis=2))] = [0, 0, 255]
            edges = cv2.Canny(red_isolated, 70, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=j, minLineLength=50, maxLineGap=10)
            if SHOW_MIDDLE_STEPS:
                print(f"Threshold: {j}, lines: {0 if lines is None else len(lines)}")
            if lines is not None:
                break
        if lines is None:
            for j in range(1,3): # No lines detected?
                # Strategy 1, dilate the red area to make the lines more visible
                kernel = np.ones((3, 3), np.uint8)
                red_isolated = cv2.dilate(red_isolated, kernel, iterations=j)
                edges = cv2.Canny(red_isolated, 70, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
                if SHOW_MIDDLE_STEPS:
                    showImage(f'Dilatation number {j}.',red_isolated)
            if lines is None: # No lines yet detected? Debug time!
                showImage('No lines detected on isolated red component.',red_isolated)
                logData(image_input)
                raise Exception("No lines detected on isolated red component.")
    
    seconds = int((( getAngleFromCenter(lines,center) +90) % 360) / 6)

    if SHOW_MIDDLE_STEPS:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(red_isolated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(lines)
        showImage('Line detected on isolated red component, shown in green.',red_isolated)
        
    #######################################################################################
    ###### Minutes pointer:
    # Convert the red isolated image to grayscale and threshold the gray_red_isolated image 
    # to get a binary mask
    gray_red_isolated = cv2.cvtColor(red_isolated, cv2.COLOR_BGR2GRAY)
    _, red_mask_binary = cv2.threshold(gray_red_isolated, 1, 255, cv2.THRESH_BINARY)
    
    if SHOW_MIDDLE_STEPS:
        showImage('Black and white with seconds pointer removed.',black_and_white_image)
    
    global last_arc
    minutes_ang=-1
    radius1 = 145          # Example radius for the first circle
    radius2 = 175          # Example radius for the second circle
    current_arc = get_arc(black_and_white_image, center, radius1, radius2)
    if last_arc is not None:
        minutes_ang = get_current_minute_pointer_angle(last_arc, current_arc, image, center)
        print(minutes_ang)
        
    # Overlay the red isolated content as white pixels on the black and white image
    black_and_white_image[red_mask_binary == 255] = 255 # basically, original image in bw w/o seconds pointer
    if last_arc is not None:
        # remove center and numbers, keeping only part of the pointers
        # -> only executed if minutes angle is already known. (min & hour pointers will have same length in the cropped img)
        r1=100
        mask = cv2.circle(np.zeros_like(black_and_white_image), center, r1, 255, -1)
        cropped = cv2.bitwise_and(black_and_white_image, black_and_white_image, mask=mask)
        outer_mm = cv2.circle(np.ones_like(black_and_white_image) * 255, center, r1, 0, -1) # adjust value subtracted to large_radius
        
        white_circle = np.zeros_like(black_and_white_image) # full black image
        cv2.circle(white_circle, center, 20, 255, -1) # draw white circle
        
        black_and_white_image = cv2.bitwise_or(cropped, cv2.bitwise_or(outer_mm, white_circle))
        if SHOW_MIDDLE_STEPS:
            showImage('Cropped', cropped)
            showImage('bw image cropped', black_and_white_image)
    last_arc = current_arc.copy()
    
    if SHOW_MIDDLE_STEPS:
        showImage('Current arc', current_arc)
        
    
    edges = cv2.Canny(black_and_white_image, 70, 150)
    # showImage('edges', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20)
    
    # A lot of lines are detected. Duplicates are detected by comparing the angles relative
    # to the center of the clock face. Lines too distant from the clock center are discarded.
    # 'too distant' = radius/2
    closest_lines = []
    angles = []
    i=0
    dof=3 # degrees of freedom, lines with angles within dof degrees of each other are considered the same
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # h = sqrt( c^2 + c1^2)
        d1=np.sqrt( (x1-center[0])**2 + (y1-center[1])**2 )
        d2=np.sqrt( (x2-center[0])**2 + (y2-center[1])**2 )
        if (np.sqrt( (x1-x2)**2 + (y1-y2)**2 )) > 183: # ignore lines too long
            continue
        ang=400
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
        else:
            # point too far, ignore it.
            #print(" - Unhandled case!! - ")
            continue
        
        if len(angles) > 0: # angles detected?
            if angles[0]+dof > ang and angles[0]-dof < ang: # this angle is about the same as the existing one?
                continue
            elif len(angles) > 1:
                if angles[1]+dof > ang and angles[1]-dof < ang:
                    continue
                elif ang != 400 and ang != minutes_ang:
                    print(f"Unexpected third angle!!! {ang}")
                    print(f"ang1: {angles[0]}, ang2: {angles[1]}")
                    print(f"last_angle_hour: {last_angle_hour}, last_angle_min: {last_angle_min}")
                    print(f"line: {line}")
                    x1, y1, x2, y2 = line[0]
                    img=image.copy()
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    showImage('Detected Lines on Black and White Image', img)
                    showImage('Black and white with seconds pointer removed.',black_and_white_image)
                    logData(image_input)
                    raise Exception("Unexpected third angle detected!! Exported troublesome frame.")
                    continue
            elif ang != 400 and ang != minutes_ang:
                if is_hm_angle_expected(ang):
                    angles.append(ang)
                    closest_lines.append(line)
                else:
                    print(f"Unexpected angle detected!! {ang:.1f}, expected near: {last_angle_hour:.1f} or {last_angle_min:.1f}")
        elif ang != 400 and ang != minutes_ang:
            if is_hm_angle_expected(ang):
                angles.append(ang)
                closest_lines.append(line)
            else:
                print(f"Unexpected angle detected!! {ang:.1f}, expected near: {last_angle_hour:.1f} or {last_angle_min:.1f}")
        i+=1

    minutes = 0
    hours = 0
    if not closest_lines:
        logData(image_input)
        raise Exception("No valid lines detected for minute and hour hands!! Exported troublesome frame.")
    if minutes_ang == -1:
        x1, y1, x2, y2 = closest_lines[0][0]
        d0=(np.sqrt( (x1-x2)**2 + (y1-y2)**2 ))
        if len(closest_lines) < 2:  # only one line detected? Hour and minutes must be overlapping
            closest_lines.append(closest_lines[0])
            if len(angles) < 2:
                angles.append(angles[0])
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
            # store_last_angle_hour_min(angles[0], angles[1], hours, minutes, seconds) # memory not implemented
        else:
            minutes = int((angles[1] % 360) / 6)
            hours = int((angles[0] % 360) / 30)
            # store_last_angle_hour_min(angles[0], angles[1], hours, minutes, seconds) # memory not implemented
    else:
        minutes = int((minutes_ang % 360) / 6)
        # print(f"Minutes angle: {minutes_ang}", end=' ')
        # print(f"Minutes: {minutes}")
        hours = int((angles[0] % 360) / 30)
        print(f"Hours angle: {angles[0]:.1f}, hour: {hours}")
            
    if SHOW_MIDDLE_STEPS:
        print(f"Time: {hours}:{minutes}:{seconds}")

    if SHOW_MIDDLE_STEPS:
        image_copy = image.copy()
        if closest_lines is not None:
            for line in closest_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        showImage('Detected Closest Lines on Clock Face', image_copy)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def processVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    e=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: # Frame not OK ?
            break
        print(f"-> Processing frame {e}")
        # if e==3:
        #     global SHOW_MIDDLE_STEPS 
        #     SHOW_MIDDLE_STEPS = True
        detected_time = doStuff(frame)
        cv2.putText(frame, detected_time, (100, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Processed Frame', frame)
        with open('detected_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'Frame {e}', detected_time])
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        e=e+1
    cap.release()
    cv2.destroyAllWindows()
    

if os.path.exists('detected_times.csv'):
    os.remove('detected_times.csv')
video_path = 'data/clock.mp4'
processVideo(video_path)
