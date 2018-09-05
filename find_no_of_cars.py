import numpy as np
import cv2

image = cv2.imread("cars_paint.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)


edged = cv2.Canny(gray, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0


for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    total += 1

print ("Found {0} objects in that image".format(total))
cv2.imshow("Output", image)
cv2.waitKey(0)
