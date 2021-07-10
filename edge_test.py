import cv2
#import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt

'''
Copied -1169337114587240122.jpg (from obverse/ms60/) as test_image.jpg
and wrote this script to automatically detect the coin edge and
crop accordingly.
'''

def chooseCenterCircle(circles, img):
    height, width = img.shape
    x_center = int(0.5*width)
    y_center = int(0.5*height)
    dist_from_center = np.zeros(len(circles[0,:]))
    for i, (x, y, r) in enumerate(circles[0,:]):
        dist_from_center[i] = np.sqrt((x-x_center)**2 + (y-y_center)**2)
    center_circle = circles[0, np.argmin(dist_from_center)]
    print('center_circle = ', center_circle)
    return center_circle

img = cv2.imread('test_image.jpg',0)
cimg = cv2.imread('test_image.jpg',1)
original = img.copy()

circle_mask = np.zeros(original.shape, dtype=np.uint8) 


height, width = img.shape

#img = cv2.imread('4186868088805551409.jpg',0)
#edges = cv2.Canny(img,20,500)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.savefig("edge_test_min20_max500")


img = cv2.medianBlur(img,5)
#gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

scale_factor = 0.25
dp = 1
#mindist = 150
mindist = int(np.round(scale_factor*np.min([width, height])))
param1 = 300
param2 = 70
#minRadius = 200
minRadius = int(np.round(scale_factor*np.min([width, height])))
maxRadius = 0

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp,mindist,
                           param1=param1,param2=param2,
                           minRadius=minRadius,maxRadius=maxRadius)

#circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp,mindist,
#                           param1=param1,param2=param2,
#                           minRadius=minRadius,maxRadius=maxRadius)

circles = np.uint16(np.around(circles))
center_circle = chooseCenterCircle(circles, img)
x = center_circle[0]
y = center_circle[1]
r = center_circle[2]
plt.figure()

# draw the outer circle
cv2.circle(cimg, (x, y), r,(0,0,0), 0)
# draw the center of the circle
#cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
# create mask
cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
img_mask = cv2.bitwise_and(original, circle_mask)
img_cropped = img_mask[(y-r):(y+r), (x-r):(x+r)]
plt.imshow(cimg)
plt.savefig("circle_test")

#img_cropped = cv2.bitwise_and(original, circle_mask)
plt.figure()
plt.imshow(img_cropped)
plt.savefig("crop_test")

plt.figure()
plt.imshow(circle_mask)
plt.savefig("mask_test")
print(circle_mask)

mask3 = cv2.cvtColor(circle_mask, cv2.COLOR_GRAY2BGR)
img_mask = cv2.bitwise_and(cimg, mask3)
img_cropped = img_mask[(y-r):(y+r), (x-r):(x+r)]
plt.figure()
plt.imshow(img_cropped)
plt.savefig("crop_test3")

