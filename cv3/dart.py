import cv2
import numpy as np
import os
import pyautogui
import keyboard 
import time

template = cv2.imread('template.png', 0)
print("Start...")

while True:

    if keyboard.is_pressed("space"):
        print("Stop...")
        break

    screenshot = pyautogui.screenshot()
    screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    
    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    template_height, template_width = template.shape
    center_x = max_loc[0] + template_width // 2
    center_y = max_loc[1] + template_height // 2

    if(max_val > 0.5):
        print(f"Target found at {(center_x, center_y)} with confidence {max_val}")
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        time.sleep(0.3)