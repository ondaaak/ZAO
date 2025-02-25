import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
matplotlib.use("WebAgg")

if not os.path.exists("cropped"):
    os.makedirs("cropped")

#cv.namedWindow("img", 0)

image1 = cv.imread("test_01.jpg", 1)
image2 = cv.imread("test_02.jpg", 1)
image3 = cv.imread("test_03.jpg", 1)
image4 = cv.imread("test_04.jpg", 1)

images = [image1, image2, image3, image4]

width = 140
height = 220
startx = 778
starty = 725

cropped_images = []

for i, img in enumerate(images, start=1):
    for j in range(3):
        x = startx + j * width
        y = starty
        block = img[y:y+height, x:x+width].copy()
        
        cv.imwrite(f"cropped/{i-1}{j+1}.png", block)
        cropped_image = cv.imread(f"cropped/{i-1}{j+1}.png")
        cropped_images.append(cropped_image)


#cary
for img in images:
    for j in range(3):
        x = startx + j * width
        y = starty
        cv.line(img, (x, y), (x + width, y + height), (0, 0, 255), 3)
        cv.line(img, (x + width, y), (x, y + height), (0, 0, 255), 3)

resized_images = [cv.resize(img, (600, 300)) for img in images]

row1 = cv.hconcat(resized_images[:2])
row2 = cv.hconcat(resized_images[2:])
combined_image = cv.vconcat([row1, row2])

cv.imshow("combined_image", combined_image)



#cropped
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
fig.suptitle('Cropped Images')

for i in range(4):
    for j in range(3):
        axs[i, j].imshow(cv.cvtColor(cropped_images[i * 3 + j], cv.COLOR_BGR2RGB))
        axs[i, j].axis('off')

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()