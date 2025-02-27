import cv2
import numpy as np
import os

output_folders = {"out-green","out-red"}

for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

green_low = np.array([40, 80, 80])
green_upp = np.array([100, 255, 255])

red_low = np.array([168, 100, 100])
red_upp = np.array([180, 255, 255])

for filename in os.listdir("test-images"):
    filepath = os.path.join("test-images", filename)

    img = cv2.imread(filepath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    greenMask = cv2.inRange(hsv, green_low, green_upp)
    resG = cv2.bitwise_and(img, img, mask=greenMask)
    
    redMask = cv2.inRange(hsv, red_low, red_upp)
    resR = cv2.bitwise_and(img, img, mask=redMask)

    if np.sum(greenMask) > np.sum(redMask):
        final_img = cv2.hconcat([img, resG, resR])
        cv2.imwrite(f"out-green/{filename}", final_img)
    elif np.sum(redMask) > np.sum(greenMask):
        final_img = cv2.hconcat([img, resR, resG])
        cv2.imwrite(f"out-red/{filename}", final_img)

print("done")