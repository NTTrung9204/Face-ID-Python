import cv2
import time
import requests
import numpy as np
import imutils
from face_id_model import FaceIDModel

ESP32_IP = "http://192.168.248.185"
CAPTURE_URL = f"{ESP32_IP}/capture"

my_model = FaceIDModel("model/detection_model.pt", "identity")

time.sleep(2.0)

while True:
    try:
        response = requests.get(CAPTURE_URL, timeout=5)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Không thể giải mã hình ảnh từ ESP32-CAM.")
                continue

            frame = imutils.resize(frame, width=600)

            identified_faces = my_model.query(frame)

            for label, face, distance in identified_faces:
                x1, y1, x2, y2 = map(int, face)

                if distance > 0.25:
                    label = "Unknow"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f"Distance: {distance:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            print(f"Lỗi {response.status_code}: Không thể lấy ảnh!")

    except Exception as e:
        print("Lỗi kết nối:", e)

    time.sleep(0.1)

cv2.destroyAllWindows()
