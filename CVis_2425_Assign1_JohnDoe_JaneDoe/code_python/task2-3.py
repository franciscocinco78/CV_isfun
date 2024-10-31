import cv2
import numpy as np
import csv
import os
import tkinter as tk
from tkinter import messagebox

# Set to True to show intermediate steps of the process, and show extra information
SHOW_MIDDLE_STEPS = False
DEBUG_FRAME = False

# center of the clock face, will not be calculated each frame, as it is assumed to be constant
center = None 
radius = 0  # same situation as 'center'
last_angle_sec = 400
last_angle_min = 400
last_angle_hour = 0
last_hour = 0
last_min = 0
last_sec = 60
error_count = 0
last_arc = None

# useful fuction to log relevant data when a critial error occurs
def logData(img,name='troublesome_frame.png'):
    global center, radius
    print(f"center: {center}")
    print(f"radius: {radius}")
    if not DEBUG_FRAME:
        cv2.imwrite(name, img)

# @todo remove this function and respective calls.
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
def getAngleFromCenter(lines, cc):
    global last_angle_sec
    angle = 400
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1 = np.sqrt((x1 - cc[0])**2 + (y1 - cc[1])**2)
        dist2 = np.sqrt((x2 - cc[0])**2 + (y2 - cc[1])**2)
        if dist1 > dist2:
            farthest_point = (x1, y1)
        else:
            farthest_point = (x2, y2)
        ang = np.degrees(np.arctan2(farthest_point[1] - cc[1], farthest_point[0] - cc[0]))+90
        if ang < 0:
            ang = 360 + ang    
        if angle != 400 and ang < last_angle_sec+15:
            angle = ang
        else:
            angle = ang
        if SHOW_MIDDLE_STEPS:
            print('Parsing ang:', ang, ', for line: ', line, ', x1: ',x1, ', y1: ',y1,', x2: ',x2, ', y2:',y2, end='; ')
    if SHOW_MIDDLE_STEPS:
        print()
    # x1, y1, x2, y2 = line[0][0]
    # dist1 = np.sqrt((x1 - _center[0])**2 + (y1 - _center[1])**2)
    # dist2 = np.sqrt((x2 - _center[0])**2 + (y2 - _center[1])**2)
    # if dist1 > dist2:
    #     farthest_point = (x1, y1)
    # else:
    #     farthest_point = (x2, y2)
    # angle = np.degrees(np.arctan2(farthest_point[1] - _center[1], farthest_point[0] - _center[0]))
    # return angle
    last_angle_sec = angle
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
# red_isolated: seconds pointer isolated, used to remove it from the 'final' difference image
def get_current_minute_pointer_angle(cc_img1, cc_img2, orig_img, cc, red_isolated):
    difference = cv2.absdiff(cc_img1, cc_img2)
    SHOW_ALL = SHOW_MIDDLE_STEPS
    if SHOW_ALL:
        cv2.imshow('get minutes -> cc_img1 (last arc)', cc_img1)
        cv2.imshow('get minutes -> cc_img2 (current arc)', cc_img2)
        showImage('get minutes -> Difference', difference)
        
    # Remove single pixels without neighbors
    kernel = np.ones((3, 3), np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
    
    # if SHOW_ALL:
    #     showImage('get minutes -> Difference clean', difference)
    # # Remove the seconds pointer from the difference image
    # gg = cv2.cvtColor(red_isolated, cv2.COLOR_BGR2GRAY)
    # # Convert the grayscale image to pure black and white, ignore returned threshold value
    # _, seconds_bw = cv2.threshold(gg, 70, 255, cv2.THRESH_BINARY)
    # # Dilate the seconds_bw image by 2 pixels
    # kernel = np.ones((3, 3), np.uint8)
    # seconds_bw = cv2.dilate(seconds_bw, kernel, iterations=3) # 3 iterations to fully remove traces of seconds pointer
    # # This may cause innacurate minutes tracking in future, must be tested.

    # if SHOW_ALL:
    #     cv2.imshow('get minutes -> Difference with seconds pointer', difference)
    # # Overlay the seconds_bw content as black pixels on the difference image
    # difference[seconds_bw == 255] = 0
    # if SHOW_ALL:
    #     showImage('get minutes -> Difference without seconds pointer', difference)
    #     showImage('get minutes -> input red_isolated', red_isolated)
    
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
            if ang < 0:
                ang = 360 + ang
            angles.append(ang)
            # print(f"Center of mass: ({cX}, {cY}), angle: {ang:.1f}")
            cv2.circle(highlighted_image, (cX, cY), 5, (255, 0, 0), -1)  # Draw
    if SHOW_ALL:
        showImage('get minutes -> highlighted_image', highlighted_image)
    if len(angles) > 1:
        if max(angles) > 350 and min(angles) < 10: # pointer crossed 00
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
        closest_angle = min(angles_copy, key=lambda x: abs(x - max(angles))) # closest angle to the max
        if SHOW_ALL:
            print(f"Angles: {angles}, max: {max(angles)}, min: {min(angles)}, closest: {closest_angle}")
        if closest_angle > (max(angles)-3):   # se os angulos forem proximos
            if SHOW_ALL:
                print("Returning average angle for minutes pointer")
            return (closest_angle + max(angles)) / 2 # devolver a media
        else:
            if SHOW_ALL:
                print("Returning max angle for minutes pointer")
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


    # Apply Gaussian blur to the isolated clock face
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    KERNEL = np.array([[0, 0, 0], [0, 1.5, 0], [0, 0, 0]])
    blurred_image = cv2.filter2D(blurred_image, -1, KERNEL)
    if SHOW_MIDDLE_STEPS:
        showImage('Blurred Clock Face', blurred_image)

    # Update the image variable to the blurred clock face
    image = blurred_image
    
    
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
        
    if DEBUG_FRAME:
        center = (int(479), int(269))
        radius = int(217)
        
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
    red_isolated_original = red_isolated.copy()
    if SHOW_MIDDLE_STEPS:
        showImage('HSV Red component Isolated', red_isolated)

    edges = cv2.Canny(red_isolated, 70, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if SHOW_MIDDLE_STEPS:
        if lines is None:
            print("Isolated red (seconds pointer) -> No lines detected at first.")
        showImage('Isolated red (seconds pointer) -> Edges.',edges)
    if lines is None:
        for j in range(90,49,-10): # No lines detected?
            # Strategy 1, make all pixels solid red and reduce the threshold to detect lines
            red_isolated[np.where((red_isolated != [0, 0, 0]).all(axis=2))] = [0, 0, 255]
            edges = cv2.Canny(red_isolated, 70, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=j, minLineLength=50, maxLineGap=10)
            if SHOW_MIDDLE_STEPS:
                print(f"Threshold: {j}, lines: {0 if lines is None else len(lines)}")
            if lines is not None:
                break
        if lines is None:
            if SHOW_MIDDLE_STEPS:
                print("Isolated red (seconds pointer) -> No lines detected at 2nd iteration.")
            for j in range(1,3): # No lines detected?
                # Strategy 2, dilate the red area to make the lines more visible
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
    if SHOW_MIDDLE_STEPS:
        img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        showImage('Detected Lines on Isolated Red Component', img)
    seconds = int((( getAngleFromCenter(lines,center) ) % 360) / 6)

    if SHOW_MIDDLE_STEPS:
        red_isolated_demo = red_isolated.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(red_isolated_demo, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"Coordinates of detected lines in seconds pointer: {lines}")
        showImage('Line detected on isolated red component, shown in green.',red_isolated_demo)
        
    #######################################################################################
    ###### Minutes pointer:
    # Convert the red isolated image to grayscale and threshold the gray_red_isolated image 
    # to get a binary mask
    gray_red_isolated = cv2.cvtColor(red_isolated_original, cv2.COLOR_BGR2GRAY)
    _, red_mask_binary = cv2.threshold(gray_red_isolated, 1, 255, cv2.THRESH_BINARY)
    
    if SHOW_MIDDLE_STEPS:
        showImage('Black and white with seconds pointer removed.',black_and_white_image)
    
    global last_arc
    minutes_ang=-1
    radius1 = 145          # Example radius for the first circle
    radius2 = 175          # Example radius for the second circle
    current_arc = get_arc(black_and_white_image, center, radius1, radius2)
    if last_arc is not None:
        minutes_ang = get_current_minute_pointer_angle(last_arc, current_arc, image, center, red_isolated)
        # print(minutes_ang)
        
        
    if DEBUG_FRAME:
        minutes_ang = 197.4656965024109
    # cv2.imshow('BW image before subtraction', black_and_white_image)
    
    # Overlay the red isolated content as white pixels on the black and white image
    black_and_white_image[red_mask_binary == 255] = 255 # basically, original image in bw w/o seconds pointer
    # Remove lone pixels without neighbors from black_and_white_image
    kernel = np.ones((3, 3), np.uint8)
    black_and_white_image = cv2.morphologyEx(black_and_white_image, cv2.MORPH_OPEN, kernel)
    # showImage('BW image after subtraction', black_and_white_image)
    
    if True: # last_arc is not None: # 
        # remove center and numbers, keeping only part of the pointers
        # ~~~~-> only executed if minutes angle is already known. (min & hour pointers will have same length in the cropped img)~~~~
        # always crop the image.
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
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=20)
    
    # if SHOW_MIDDLE_STEPS:
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detected Lines on Black and White Image', image)
    
    if lines is None:
        showImage('No lines detected on isolated black and white component.',edges)
        raise Exception("No lines detected on isolated black and white component.")
    
    # A lot of lines are detected. Duplicates are detected by comparing the angles relative
    # to the center of the clock face. Lines too distant from the clock center are discarded.
    # 'too distant' = radius/2
    closest_lines = []
    angles = []
    i=0
    dof=3 # degrees of freedom, lines with angles within dof degrees of each other are considered the same
    global last_hour
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # h = sqrt( c^2 + c1^2)
        d1=np.sqrt( (x1-center[0])**2 + (y1-center[1])**2 )
        d2=np.sqrt( (x2-center[0])**2 + (y2-center[1])**2 )
        if (np.sqrt( (x1-x2)**2 + (y1-y2)**2 )) > 183: # ignore lines too long
            if SHOW_MIDDLE_STEPS:
                print('Ignored line (too long):', line, end='; ')
            continue
        ang=400
        # a line has 2 points
        if d1 < radius/2 and d1 < d2:   # point 1 is closer to the center
            x2+= -1*(x1-center[0]) # move the line to the center, 
            y2+= -1*(y1-center[1]) # just so angles are consistent between iterations
            ang= np.degrees(np.arctan2(y2 - center[1], x2 - center[0])) + 90
        elif d2 < radius/2 and d2 < d1: # point 2 is closer to the center
            x1+= -1*(x2-center[0]) 
            y1+= -1*(y2-center[1])
            ang= np.degrees(np.arctan2(y1 - center[1], x1 - center[0])) + 90
        else:
            if SHOW_MIDDLE_STEPS:
                print('Ignored faraway point:', line, ' with distances: ', d1, ' ', d2, end='; ')
            continue
        if ang < 0:
            ang += 360
        if SHOW_MIDDLE_STEPS:
            print('Parsing ang:', ang, ', for line: ', line, ', x1: ',x1, ', y1: ',y1,', x2: ',x2, ', y2:',y2, end='; ')
            
            img=image_input.copy()
            cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
            showImage('Processing this line', img)
            img=image_input.copy()
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            showImage('Adjusted far point in line', img)
            
        if len(angles) > 0: # already detected angles?
            # Minutes pointer is faster than hours, before 7 hours, minutes pointer is the first to be detected (lower angle),
            # after 6 hours, hours pointer is the first to be detected (lower angle). 
            if (angles[0]+dof > ang and angles[0]-dof < ang) and ang > angles[0]: # and last_hour<6: # this angle is about the same as the existing one, but higher?
                angles[0] = ang # replace angle
                continue # finish iteration
            # elif (angles[0]+dof > ang and angles[0]-dof < ang) and ang < angles[0] and last_hour>=6: # this angle is about the same as the existing one, but lower?
            #     angles[0] = ang # replace angle
            #     continue # finish iteration
            elif angles[0]+dof > ang and angles[0]-dof < ang: # this angle is about the same as the existing one?
                continue # do not add it
            elif len(angles) > 1:
                if (angles[1]+dof > ang and angles[1]-dof < ang) and ang > angles[1] and last_hour<6: # this angle is about the same as the existing one, but higher?
                    angles[1] = ang # replace angle
                    continue # finish iteration
                # elif (angles[1]+dof > ang and angles[1]-dof < ang) and ang < angles[1]: # this angle is about the same as the existing one, but higher?
                #     angles[1] = ang # replace angle
                #     continue # finish iteration
                if angles[1]+dof > ang and angles[1]-dof < ang:
                    continue
                elif ang != 400 and ang != minutes_ang:
                    # print(f"Unexpected third angle!!! {ang}")
                    # print(f"ang1: {angles[0]}, ang2: {angles[1]}")
                    # print(f"last_angle_hour: {last_angle_hour}, last_angle_min: {last_angle_min}")
                    # print(f"line: {line}")
                    # x1, y1, x2, y2 = line[0]
                    # img=image.copy()
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # showImage('Detected Lines on Black and White Image', img)
                    # showImage('Black and white with seconds pointer removed.',black_and_white_image)
                    # logData(image_input)
                    # raise Exception("Unexpected third angle detected!! Exported troublesome frame.")
                    continue
            elif ang != 400 and ( not (minutes_ang-dof < ang < minutes_ang + dof)) and minutes_ang+dof < 360 and minutes_ang-dof > 0: # ensure it is not the minutes angle
                angles.append(ang)
                closest_lines.append(line)
            elif ang != 400 and ( not (minutes_ang-dof < ang < (minutes_ang +dof -360)) and minutes_ang+dof > 360): #ensure it is not the minutes angle
                angles.append(ang)
                closest_lines.append(line)
            elif ang != 400 and ( not (minutes_ang-dof+360 < ang < (minutes_ang +dof+360)) and minutes_ang-dof < 0): # ang=359.0, min_ang=1.0
                angles.append(ang)
                closest_lines.append(line)
        elif ang != 400: # no special checks, first found angle can be any
            angles.append(ang)
            closest_lines.append(line)
        i+=1
    # print(angles)
    minutes = 0
    hours = 0
    hours_angle = 400
    if SHOW_MIDDLE_STEPS:
        print(f"Hours Angles: {angles}")
        print(f"Minutes angle: {minutes_ang}")
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
            # hours = int((angles[1] % 360) / 30)
            hours_angle = angles[1]
        else:
            minutes = int((angles[1] % 360) / 6)
            # hours = int((angles[0] % 360) / 30)
            hours_angle = angles[0]
        hours = int((hours_angle % 360) / 30)
    else:
        minutes = int((minutes_ang % 360) / 6)
        
        if len(angles) < 1:
            print('Minuts angle: ', minutes_ang)
            logData(image_input)
            raise Exception("No angles obtained! Exported troublesome frame.")
        if len(angles) < 2:
            hours_angle = angles[0]
            # hours = int((angles[0] % 360) / 30)
        else:
            if ((minutes_ang-6 < angles[0] < minutes_ang + 6) ): #and (minutes_ang+6 < 360)): # angle[0] is the minute hand ?
                # hours = int((angles[1] % 360) / 30) # yes, use angle[1] as hour hand
                hours_angle = angles[1]
            elif ((angles[0] < (minutes_ang +6 -360)) and (minutes_ang+6 > 360)): # in case min is 359 and hours 0.0
                hours_angle = angles[1]
            else:
                # hours = int((angles[0] % 360) / 30)
                hours_angle = angles[0]
        hours = int((hours_angle % 360) / 30)
        if SHOW_MIDDLE_STEPS:
            if len(angles) > 1:
                print(f"Minutes angle: {minutes_ang:.1f}, angles: {angles[0]:.1f}, {angles[1]:.1f}, hour: {hours}, len(angles): {len(angles)}")
            else:
                print(f"Minutes angle: {minutes_ang:.1f}, hours angle: {angles[0]:.1f}, hour: {hours}, len(angles): {len(angles)}")
        # print(f"Hours angle: {angles[0]:.1f}, hour: {hours}")
    global last_min
    if hours == last_hour+1 and minutes > 56: # smart-fix 'clock transitions
        hours -= 1
    elif hours == last_hour-1 and minutes < 5:
        hours +=1
    elif hours == last_hour-1 and (last_angle_hour+1 > hours_angle > last_angle_hour-1): # allow 1 deg error
        hours +=1
    elif hours == 0 and last_hour == 11 and minutes > 56:
        hours = 11
    
    if hours != last_hour and hours != last_hour+1:
        if not (last_hour == 11 and hours == 0):
            if len(angles) > 1:
                print(f"Minutes: {minutes}, minutes angle: {minutes_ang:.1f}, last_min: {last_min}, last_ang_min: {last_angle_min}")
                print(f"angles: {angles[0]:.1f}, {angles[1]:.1f}, hour: {hours}, len(angles): {len(angles)}, hours_angle: {hours_angle:.1f}, last_angle_hour: {last_angle_hour:.1f}")
            else:
                print(f"Minutes: {minutes}, minutes angle: {minutes_ang:.1f}, last_min: {last_min}, last_ang_min: {last_angle_min}")
                print(f"angles: {angles[0]:.1f}, hour: {hours}, len(angles): {len(angles)}, hours_angle: {hours_angle:.1f}, last_angle_hour: {last_angle_hour:.1f}")
            logData(image_input)
            print(f"Hours: {hours}, last_hour: {last_hour}")
            raise Exception("Inconsistent hours detected!! Expor ted troublesome frame.")
    else:
        # logData(image_input,'prevoius_frame.png')
        last_hour = hours
        last_angle_hour = hours_angle
    
    if minutes == last_min-1:
        minutes +=1
    if (minutes != last_min and minutes != last_min+1 and last_min != 59) or (minutes == 0 and minutes != last_min and last_min != 59):
        print(f"Minutes: {minutes}, minutes angle: {minutes_ang:.1f}, last_angle_min: {last_angle_min:.1f}")
        logData(image_input)
        print(f"Minutes: {minutes}, last_min: {last_min}")
        raise Exception("Inconsistent minutes detected!! Exported troublesome frame.")
    else:
        last_min = minutes
        last_angle_min = minutes_ang
        
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

########################################################################################

def processVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    e=1
    global SHOW_MIDDLE_STEPS 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: # Frame not OK ?
            break
        print(f"-> Processing frame {e}")
        # if e==202:
        #     SHOW_MIDDLE_STEPS = True
        # else:
        #     SHOW_MIDDLE_STEPS = False
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
    

if DEBUG_FRAME  ==  False:  # Normal execution
    if os.path.exists('detected_times.csv'):
        os.remove('detected_times.csv')
    video_path = 'data/clock.mp4'
    processVideo(video_path)
else:   # Debug frame
    SHOW_MIDDLE_STEPS = True
    image_path = 'troublesome_frame.png'
    image = cv2.imread(image_path)
    detected_time = doStuff(image)
    print(f"Detected time: {detected_time}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()