import cv2

cap = cv2.VideoCapture("datasets/busy_parking_lot.mp4")

ret, frame = cap.read()

print(ret)
print(frame.shape)

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()