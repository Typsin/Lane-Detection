import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest (img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def Line_drawing (img, lines):
    img = np.copy(img)
    canvas_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines: 
        for x1, y1, x2, y2 in line:
            cv2.line(canvas_image, (x1, y1), (x2, y2), (255, 0, 255), thickness=10)
    
    img = cv2.addWeighted(img, 0.8, canvas_image, 1, 0.0)
    #o.g image, the weight, paste image to, beta value,gamma value
    return img

image = cv2.imread('road_lane.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (0,870),
    (960, 475),
    (1920,950),
    (width, height)
]

#gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#canny_image = cv2.Canny(gray_image, 100, 200)
#cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)

cropped_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32),)
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 500)
lines = cv2.HoughLinesP(canny_image, rho=6 , theta=np.pi/60, threshold=100, lines= np.array([]), minLineLength=40, maxLineGap=25)


line_drawn = Line_drawing(image, lines)
#plt.imshow(cropped_image)
#plt.imshow(canny_image)
plt.imshow(line_drawn)
plt.show()