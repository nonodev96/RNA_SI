import cv2
import numpy as np
import os

CURRENT_WORKING_DIRECTORY = os.getcwd()

img = cv2.imread(CURRENT_WORKING_DIRECTORY + '/examples/data/hu-moments-shape.png')
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,170,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:",len(contours))

# compute HuMoments for all the contours detected in the image
for i, cnt in enumerate(contours):
   x,y = cnt[0,0]
   moments = cv2.moments(cnt)
   hm = cv2.HuMoments(moments)
   cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
   cv2.putText(img, f'Contour {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
   print(f"\nHuMoments for Contour {i+1}:\n", hm)

cv2.imshow("Hu-Moments", img)
cv2.waitKey(0)
cv2.destroyAllWindows()