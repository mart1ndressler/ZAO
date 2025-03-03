import cv2 as cv
import numpy as np
import os

os.makedirs("out-green", exist_ok=True)
os.makedirs("out-red", exist_ok=True)

for fname in os.listdir("test-images"):
    img = cv.imread(f"test-images/{fname}")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    g_mask = cv.inRange(hsv, np.array([40, 80, 80]), np.array([100, 255, 255]))
    r_mask = cv.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | cv.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    
    green_count = cv.countNonZero(g_mask)
    red_count = cv.countNonZero(r_mask)
    out = cv.hconcat([img, cv.bitwise_and(img, img, mask=g_mask), cv.bitwise_and(img, img, mask=r_mask)])
    
    if green_count >= red_count:
        cv.imwrite(f"out-green/{fname}", out)
        print(f"{fname}: green = {green_count}, red = {red_count} -> Saved to out-green.")
    else:
        cv.imwrite(f"out-red/{fname}", out)
        print(f"{fname}: green = {green_count}, red = {red_count} -> Saved to out-red.")
print("Done, all images processed!")
