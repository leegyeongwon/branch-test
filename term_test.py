import cv2

Image = input('type the name of image : ')
face_cascade = cv2.CascadeClassifier('C:/Users/wasse/Desktop/oss_opencv/haarcascade_frontalface_default.xml')

img = cv2.imread('C:/Users/wasse/Desktop/oss_opencv/{}.jpg'.format(Image))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.2, 3)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h),(255,255,255),2)

cv2.imshow('img',img)
cv2.waitKey()