import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import time

ESP32_IP = "http://192.168.1.5"
CAPTURE_URL = f"{ESP32_IP}/capture"

plt.ion()
fig, ax = plt.subplots()

while True:
    try:
        response = requests.get(CAPTURE_URL, timeout=5)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.clear()
            ax.imshow(img)
            ax.axis("off")
            plt.draw()
            plt.pause(0.1)
        else:
            print(f"Lỗi {response.status_code}: Không thể lấy ảnh!")
    
    except Exception as e:
        print("Lỗi kết nối:", e)
    
    time.sleep(0.5)
