import cv2
import os
import numpy as np

image_path  = "/Users/niharikasoni/Desktop/images/"
output_path = "/Users/niharikasoni/Desktop/output/"
haar_cascade = cv2.CascadeClassifier('/Users/niharikasoni/Downloads/Haarcascade_frontalface_default.xml')
image_list = os.listdir(image_path)
print(image_list)

for count,image in enumerate(image_list):
    print("count",count)

    image_read = cv2.imread(image_path + image)
    height = 224
    width = image_read.shape[1]*height/image_read.shape[0]
    image_read = cv2.resize(image_read, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("image_read",image_read)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    for coordinate in faces_rect:
        (x, y, w, h) = coordinate
        # colors = np.random.randint(1, 255, 3)
        roi_color = image_read[y:y + h, x:x + w]
        cv2.rectangle(image_read, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(output_path + str(w) + str(h) + "face_" + str(count) + '.jpg', roi_color)
    cv2.imshow('Image', image_read)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  



    

