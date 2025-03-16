import cv2
import time
from imutils.video import VideoStream
import imutils
from face_id_model import FaceIDModel

my_model = FaceIDModel("model/detection_model.pt", "identity")
esp32_url = "http://192.168.71.235:81/stream"
# vs = VideoStream(src=0).start()
vs = VideoStream(src=esp32_url).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    if frame is None:
        print("Không thể đọc khung hình từ video stream.")
        break
    
    frame = imutils.resize(frame, width=800)

    identified_faces = my_model.query(frame)

    for label, face, distance in identified_faces:
        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if distance > 0.25:
            label = "Unknow"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()