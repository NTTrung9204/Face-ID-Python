import cv2
import time
from imutils.video import VideoStream
import imutils
from face_id_model import FaceIDModel
import requests

my_model = FaceIDModel("model/detection_model.pt", "identity")
# esp32_url = "http://192.168.55.235:81/stream"
# esp32_url = "http://192.168.1.6:81/stream"
esp32_url = "http://192.168.180.235:81/stream"
# vs = VideoStream(src=0).start()
vs = VideoStream(src=esp32_url).start()
time.sleep(2.0)

index = 0

while True:
    frame = vs.read()
    if frame is None:
        print("Không thể đọc khung hình từ video stream.")
        break
    
    # frame is already in BGR format
    frame = imutils.resize(frame, width=800)

    if index % 10 == 0:

        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        response = requests.post(
            "http://localhost:5000/api/identity_student",
            files=files
        )

        print(response.status_code)
        
    identified_faces = my_model.query(frame)

    for label, face, distance, _, _ in identified_faces:
        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if distance > 0.25 and label != "Fake":
            label = "Unknow"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)


    index += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()