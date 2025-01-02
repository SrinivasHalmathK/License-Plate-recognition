import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

image = cv2.imread("C:/Users/Srinivas halmath k/OneDrive/Desktop/cg_project/Vehicle-Number-Plate-Reading/car.jpeg")
img = image

image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None 

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  
        NumberPlateCnt = approx 
        break


mask = np.zeros(image.shape, dtype=np.uint8)


cv2.drawContours(mask, [NumberPlateCnt], -1, (255, 255, 255), -1)


if NumberPlateCnt is not None:
    border_color = (0, 0, 255)  
    border_thickness = 4
    cv2.drawContours(image, [NumberPlateCnt], -1, border_color, border_thickness)

    
    mask_inv = cv2.bitwise_not(mask)

    
    image[mask_inv == 255] = 0


cv2.imshow("Orginal Image",img)
cv2.namedWindow("Final Image with Border", cv2.WINDOW_NORMAL)
cv2.imshow("Final Image with Border", image)


config = ('-l eng --oem 1 --psm 3')


x, y, w, h = cv2.boundingRect(NumberPlateCnt)
roi = image[y:y+h, x:x+w]


text = pytesseract.image_to_string(roi, config=config)


raw_data = {'date': [time.asctime(time.localtime(time.time()))], 
            'v_number': [text]}

df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
df.to_csv('data.csv')


print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()

