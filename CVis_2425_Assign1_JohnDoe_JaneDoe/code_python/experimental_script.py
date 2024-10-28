import cv2
import numpy as np

def showImage(d,i):
    cv2.imshow(d, i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

    # Remove single pixels without neighbors
    kernel = np.ones((3, 3), np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

    # Find contours of the objects in the difference image
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    highlighted_image = orig_img.copy()
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # Red contours with thickness 2
    # Find and print the center of mass of each object
    angles = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            ang = np.degrees(np.arctan2(cY - cc[1], cX - cc[0]))
            angles.append(ang)
            print(f"Center of mass: ({cX}, {cY}), angle: {ang:.1f}")
            cv2.circle(highlighted_image, (cX, cY), 5, (255, 0, 0), -1)  # Draw
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
    
# Load the image
image_path = 'data/clock.png'
image = cv2.imread(image_path)

# Define the centers and radii of the circles
center1 = (479, 269)  # Example center for the first circle
radius1 = 145          # Example radius for the first circle
radius2 = 175          # Example radius for the second circle

# Draw the circles on the image
cv2.circle(image, center1, radius1, (0, 255, 0), 2)  # Green circle with thickness 2
cv2.circle(image, center1, radius2, (255, 0, 0), 2)  # Blue circle with thickness 2

# Display the result
cv2.imshow('Image with Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, bw_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# # Create a mask with the larger circle
# mask = cv2.circle(np.zeros_like(bw_image), center1, radius2, 255, -1) # .
# # Subtract the smaller circle from the mask
# mask = cv2.circle(mask, center1, radius1, 0, -1) # O
# cropped_bw_image = cv2.bitwise_and(bw_image, bw_image, mask=mask)

# # Draw a full white circle with the radius of the small circle
# white_circle = np.zeros_like(bw_image) # full black image
# cv2.circle(white_circle, center1, radius1, 255, -1) # draw white circle
# outer_mask = cv2.circle(np.ones_like(bw_image) * 255, center1, radius2-2, 0, -1)

# # Apply the mask to the cropped binary image
# cropped_bw_image = cv2.bitwise_or(cropped_bw_image, cv2.bitwise_or(outer_mask, white_circle))
cropped_bw_image = get_arc(bw_image, center1, radius1, radius2)

# Display the cropped binary image
cv2.imshow('Cropped Binary Image', cropped_bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = cv2.imread('troublesome_frame.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, bw_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# mask = cv2.circle(np.zeros_like(bw_image), center1, radius2, 255, -1)
# mask = cv2.circle(mask, center1, radius1, 0, -1)
# cropped_bw_image2 = cv2.bitwise_and(bw_image, bw_image, mask=mask)

# # Make the area without content white
# white_background = np.ones_like(cropped_bw_image2) * 255

cv2.imshow('Second bw Image', bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped_bw_image2 = get_arc(bw_image, center1, radius1, radius2)

# Display the cropped binary image
cv2.imshow('Second Cropped Binary Image', cropped_bw_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(get_current_minute_pointer_angle(cropped_bw_image, cropped_bw_image2, image, center1))

# # Find differences between the two cropped binary images
# difference = cv2.absdiff(cropped_bw_image, cropped_bw_image2)

# # Remove single pixels without neighbors
# kernel = np.ones((3, 3), np.uint8)
# difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

# # Find contours of the objects in the difference image
# contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on the original image
# highlighted_image = image.copy()
# cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # Red contours with thickness 2
# # Find and print the center of mass of each object
# for contour in contours:
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         angle = np.degrees(np.arctan2(cY - center1[1], cX - center1[0]))
#         print(f"Center of mass: ({cX}, {cY}), angle: {angle:.1f}")
#         # print(f"Center of mass: ({cX}, {cY})")
#         cv2.circle(highlighted_image, (cX, cY), 5, (255, 0, 0), -1)  # Draw center of mass

# # Display the highlighted image
# cv2.imshow('Highlighted Objects', highlighted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Difference Image', difference)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
newRadius1 = 100
mask = cv2.circle(np.zeros_like(bw_image), center1, newRadius1, 255, -1) # .

white_circle = np.zeros_like(bw_image) # full black image
cv2.circle(white_circle, center1, 20, 255, -1) # draw white circle

cropped = cv2.bitwise_and(bw_image, bw_image, mask=mask)
outer_mm = cv2.circle(np.ones_like(bw_image) * 255, center1, newRadius1, 0, -1) # adjust value subtracted to large_radius
# Apply the mask to the cropped binary image
cropped = cv2.bitwise_or(cropped, cv2.bitwise_or(outer_mm, white_circle))
showImage('Cropped', cropped)
# Find contours of the objects in the cropped image
# Invert the cropped image
cropped = cv2.bitwise_not(cropped)

# Find contours of the objects in the inverted cropped image
contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate and print the area of each contour, and highlight the areas and center of mass
highlighted_image = image.copy()
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Area: {area}")
    
    # Draw the contour
    cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)  # Red contour with thickness 2
    
    # Calculate the center of mass
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(f"Center of mass: ({cX}, {cY})")
        
        # Draw the center of mass
        cv2.circle(highlighted_image, (cX, cY), 5, (255, 0, 0), -1)  # Blue center of mass

# Display the highlighted image
showImage('Highlighted Areas and Centers of Mass', highlighted_image)
