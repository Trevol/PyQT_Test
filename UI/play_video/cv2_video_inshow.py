import cv2

cap = cv2.VideoCapture("d:\DiskE\Computer_Vision_Task\Video 2.mp4")
delay = 1000 // 15

while 1:
    _, frame = cap.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()