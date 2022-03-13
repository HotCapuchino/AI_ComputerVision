import cv2
import matplotlib.pyplot as plt
import numpy as np


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError('Camera doesnt work')

cv2.namedWindow('binary', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)

# blue = 80 160 160
mask_colors = [(40, 100, 100), (120, 180, 180)]

while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)
    blurred = cv2.medianBlur(image.copy(), 25)
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, mask_colors[0], mask_colors[1])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (255, 0, 0), 10)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist, 0.75 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(mask, fg)

    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    markers = cv2.watershed(image, markers.copy())

    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    balls_amount = 0
    for i in range(len(contours)):
        if  hierarchy[0][i][3] == -1:
            
            if cv2.contourArea(contours[i]) < image.shape[1] * image.shape[0] * 0.9:
                balls_amount += 1
                cv2.drawContours(image, contours, i, (255, 0, 0), 10)
    
    print(balls_amount // 2)

    cv2.imshow('camera', image)
    cv2.imshow('binary', mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()