import base64
import io
import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS, cross_origin
from face_id_model import FaceIDModel
from utils.model_utils import extract_face_embedding
from utils.server_utils import ServerUtils
from PIL import Image
import numpy as np
import requests
import mysql.connector
from datetime import datetime
import websocket
import json
import threading

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
# CORS(app)
CORS(app, resources={r"/api/*": {
    "origins": "http://192.168.180.164:5173",  # Thay đổi theo origin của React app
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})


my_model = FaceIDModel("model/detection_model.pt", "identity")

results = ""

# Địa chỉ WebSocket của ESP32
ESP32_WS_URL = "ws://192.168.180.207:81"  # Thay đổi IP này thành IP của ESP của bạn

@app.route('/student_images/<filename>')
def student_images(filename):
    return send_from_directory("student_images", filename)

# Hàm gửi thông báo đến ESP32 qua WebSocket
def send_to_esp32(message_data):
    try:
        def on_open(ws):
            print("Đã kết nối tới ESP32 WebSocket")
            ws.send(json.dumps(message_data))
            print(f"Đã gửi: {message_data}")
            # Đóng kết nối sau khi gửi
            ws.close()

        def on_close(ws, close_status_code, close_msg):
            print("Đã ngắt kết nối WebSocket")

        def on_error(ws, error):
            print(f"Lỗi kết nối WebSocket: {error}")

        # Tạo và chạy WebSocket client trong thread riêng
        ws = websocket.WebSocketApp(ESP32_WS_URL,
                                    on_open=on_open,
                                    on_close=on_close,
                                    on_error=on_error)
        
        # Chạy trong thread riêng để không block server
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        return True
    except Exception as e:
        print(f"Lỗi khi gửi thông báo đến ESP32: {e}")
        return False

# Thêm route API để kiểm tra kết nối ESP32
@app.route("/api/esp32/status", methods=["GET"])
def esp32_status():
    try:
        # Gửi ping đến ESP32
        test_msg = {"type": "ping", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if send_to_esp32(test_msg):
            return {"message": "Ping sent to ESP32", "status": "success"}
        else:
            return {"message": "Failed to send ping to ESP32", "status": "error"}
    except Exception as e:
        return {"message": f"Error checking ESP32 status: {e}", "status": "error"}

# Thêm route để gửi thông báo đến ESP32 khi nhận diện thành công
@app.route("/api/esp32/notify", methods=["POST"])
def notify_esp32():
    data = request.get_json()
    message = data.get('message', 'Attendance recorded')
    student_id = data.get('student_id')
    lesson_id = data.get('lesson_id')
    name = data.get('name', '')
    
    send_data = {
        "student_id": student_id,
        "lesson_id": lesson_id,
        "name": name,
        "message": message
    }
    
    success = send_to_esp32(send_data)
    
    if success:
        return {"message": "Notification sent to ESP32", "status": "success"}
    else:
        return {"message": "Failed to send notification to ESP32", "status": "error"}

@app.route("/save_image", methods=["POST"])
def save_image():
    label = request.form.get("label")
    image = request.files.get("image")

    if not label:
        return {"message": "Label is None!", "status": "error"}

    if not image:
        return {"message": "No image received!", "status": "error"}

    print("Label:", label)
    print("Image received:", image.filename)

    try:
        ServerUtils.save_image(label, image)
        return {"message": "Save image successfully!", "status": "success"}
    except:
        return {"message": "Interal Server Error!", "status": "error"}


@app.route("/api/identity_student", methods=["POST"])
def identity_student():
    try:
        image = request.files.get("image")

        if not image:
            return {"message": "No image received!", "status": "error"}

        print("Image received:", image.filename)

        # Convert image to numpy array
        img = Image.open(image)
        img_array = np.array(img)


        # convert to BGR format
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        print("Image shape:", img_array.shape)

        student_vectors = ServerUtils.fetch_student_vectors()
        # print("Student vectors:", student_vectors)
        label, distance = ServerUtils.identify_student(
            student_vectors, img_array, my_model.extractor_model
        )

        print("Label:", label)
        print("Distance:", distance)

        if distance > 0.3:
            return {"label": "unknown", "distance": distance, "status": "success"}
        
        # save image to file
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_name = f"image_{current_time}.jpg"
        filename = f"student_images/image_{current_time}.jpg"
        img.save(filename)
        print(f"Image saved as {filename}")

        studentId = label

        student = ServerUtils.get_student_by_id(student_id=studentId)
        if not student:
            print("Student not found")
            return {"message": "Student not found", "status": "error"}
        
        label = student["name"]

        # label = str(label)

        # if label == "1":
        #     label = "trung"
        # elif label == "2":
        #     label = "tai"
        # elif label == "3":
        #     label = "hoang"
        # elif label == "4":
        #     label = "phong"

        # global results
        # results = label

        print("Student ID:", studentId)
        print("Label:", label)

        # Get current lesson for this student
        lessonId = ServerUtils.get_current_lesson_for_student(studentId)
        
        if not lessonId:
            return {"message": "Student not found in any active lessons", "status": "error"}
        
        # post data to localhost:8080/api/attendance/check
        try:
            response = requests.post(
                "http://192.168.180.164:8080/api/attendance/check",
                json={"lessonId": lessonId, "studentId": studentId, "image": image_name},
            )
        except requests.exceptions.RequestException as e:
            print("Error sending request to attendance API:", e)
            return {"message": "Error sending request to attendance API", "status": "error"}
        
        print("Status Code:", response.status_code)
        print("Response:", response.json())

        print("Lesson ID:", lessonId)
        print("Student ID:", studentId)
        
        # Thông báo cho ESP32
        notification = {
            "student_id": studentId,
            "lesson_id": lessonId,
            "name": label,
            "message": f"Attendance recorded for student {label}"
        }
        send_to_esp32(notification)

        return {"label": label, "distance": distance, "lessonId": lessonId, "status": "success"}
    except Exception as e:
        print("Error processing image:", e)
        return {"message": "Internal Server Error", "status": "error"}


@app.route("/api/result", methods=["GET"])
def get_result():
    global results
    return {"results": results, "status": "success"}

@app.route('/api/face/register', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def register_face():
    # Xử lý riêng cho OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify(success=True)
    
    # Xử lý POST request
    data = request.get_json()
    if not data:
        print("Không nhận được dữ liệu JSON")
        return jsonify(error="No data provided"), 400
    
    username = data.get('username')
    images = data.get('images')
    
    print(f"Username: {username}")
    print(f"Số lượng ảnh: {len(images) if images else 'None'}")

    array_vector = []
    
    # Xử lý logic đăng ký khuôn mặt ở đây
    for image in images:
        # image = Image.open(image)
        # read image from base64 string
        image = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
        image_array = np.array(image)

        # print(extract_face_embedding(image_array, my_model.extractor_model))
        array_vector.append(extract_face_embedding(image_array, my_model.extractor_model).tolist())

    response = requests.post(
        "http://192.168.180.164:8080/api/student-vectors",
        json={
            "username": username,
            "featureVector": array_vector,
        }
    )
    print("Status Code:", response.status_code)


    # gọi api student-vectors

    return {"message": "Register face successfully!", "status": "success"}


if __name__ == "__main__":
    # Cài đặt websocket-client nếu chưa cài: pip install websocket-client
    print("Starting server on 0.0.0.0:5000...")
    print(f"ESP32 WebSocket URL: {ESP32_WS_URL}")
    print("Server ready - waiting for connections")
    app.run(host="0.0.0.0", port=5000, debug=True)
