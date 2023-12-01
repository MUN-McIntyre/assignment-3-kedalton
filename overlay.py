import numpy as np
import cv2 as cv

img1 = cv.imread("backpack.jpg", 2) # grayscale img
img2 = cv.imread("tree.jpg", 2)
assert img1 is not None and img2 is not None, "file not found."

img1 = cv.resize(img1, (img2.shape[1], img2.shape[0])) # make sure img1 is same size as img2

# inverse binary thresholding
def inv_threshold(img, threshold, maxval):
    img_threshold = img.copy()

    for x in range(img_threshold.shape[0]):
        for y in range(img_threshold.shape[1]):
            val = img_threshold[x,y]
            if val > threshold:
                img_threshold[x,y] = 0
            else:
                img_threshold[x,y] = maxval
    return img_threshold

mask = inv_threshold(img1, 45, 255) # inverse threshold
mask = cv.medianBlur(mask, 5) # median blur for popcorn noise

backpack = cv.bitwise_and(mask, img1) # get backpack by itself using mask and img1
tree = cv.subtract(img2, mask) # make a space for the backpack to go by subtracting the mask from img2.
result = cv.add(tree, backpack) # add backpack under tree


# midpoint algo using symmetry from slides, render a white circle on result.
def render_circle(center, r):
    x = 0
    y = r
    d = 1 - r
    c_x, c_y = center
    while y >= x:
        result[c_x + x, c_y + y] = 255
        result[c_x + x, c_y - y] = 255
        result[c_x - x, c_y + y] = 255
        result[c_x - x, c_y - y] = 255

        result[c_x + y, c_y + x] = 255
        result[c_x + y, c_y - x] = 255
        result[c_x - y, c_y - x] = 255
        result[c_x - y, c_y + x] = 255

        if d < 0:
            d += (x*2) + 3
            x += 1
        else:
            d += (x - y)*2 + 5
            x += 1
            y -= 1

render_circle((423, 169), 30) # render white circle around the logo in the center of the backpack
cv.imwrite("result.jpg", result)

# show result
cv.imshow("result", result)
cv.waitKey(0)