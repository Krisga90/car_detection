import cv2


video_path: str = "video/video2.avi"
cap = cv2.VideoCapture(video_path)
faceCascade = cv2.CascadeClassifier("haarcascades/cars.xml")

while True:
    ret, frame = cap.read()

    if ret:
        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale: float = 1.1
        minimum_neighbours: int = 3
        faces = faceCascade.detectMultiScale(gray_scale_frame, scale, minimum_neighbours)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

        cv2.imshow("cammera", frame)
        if cv2.waitKey(100) and 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
