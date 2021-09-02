import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img=cv.imread("Photos/pikachu2.jpg",cv.IMREAD_COLOR)
 
# Convert to Gray image
gray= cv.cvtColor(img,cv.COLOR_BGRA2GRAY) #bgr image

# Plot histogram de tim Threshold --->thresh = 210
# hist = cv.calcHist([gray],[0],None,[256],[0,256])
# x = range(256)
# plt.plot(x, hist)
# plt.show()

# Segmentaion
thresh = 250
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]  #cai nao lon hon 120 thi la mau trang


# Morphology to seperate rice seeds
kernel = np.ones((3,3),np.uint8)
b_img = cv.erode(b_img, kernel, iterations=1)

#Find contour 
contours, hierachy = cv.findContours(b_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
print(len(contours))


for c in contours:
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow("pika", b_img)
cv.imshow("result", img)
cv.waitKey(0)           #doi trong bao lau
cv.destroyAllWindows()