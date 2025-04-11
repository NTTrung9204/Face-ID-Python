import base64
import io
import cv2
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
from face_id_model import FaceIDModel
from utils.model_utils import extract_face_embedding
from utils.server_utils import ServerUtils
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
# CORS(app)
CORS(app, resources={r"/api/*": {
    "origins": "http://localhost:5173",  # Thay đổi theo origin của React app
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})


my_model = FaceIDModel("model/detection_model.pt", "identity")

results = ""


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
    label, distance = ServerUtils.identify_student(
        student_vectors, img_array, my_model.extractor_model
    )

    if distance > 0.25:
        return {"label": "unknown", "distance": distance, "status": "success"}

    studentId = label

    label = str(label)

    if label == "1":
        label = "trung"
    elif label == "2":
        label = "tai"
    elif label == "3":
        label = "hoang"
    elif label == "4":
        label = "phong"

    global results
    results = label

    # post data to localhost:8080/api/attendance/check

    response = requests.post(
        "http://localhost:8080/api/attendance/check",
        json={"lessonId": 2, "studentId": studentId},
    )

    print("Status Code:", response.status_code)
    print("Response:", response.json())

    return {"label": label, "distance": distance, "status": "success"}


@app.route("/api/result", methods=["GET"])
def get_result():
    global results
    return {"results": results, "status": "success"}


# @app.route("/api/face/register", methods=["POST"])
# @cross_origin(supports_credentials=True)
# def register_face():
#     images = request.files.get("images")
#     username = request.form.get("username")

#     print("Username:", username)
#     print("Images received:", images)
    
#     if not username:
#         return {"message": "Username is None!", "status": "error"}
    
#     if not images:
#         return {"message": "No images received!", "status": "error"}
    
#     print(username)
#     print("Length of images:", len(images))

#     for image in images:
#         image = Image.open(image)
#         image_array = np.array(image)

#         print(extract_face_embedding(image_array, my_model.extractor_model))

#     return {"message": "Register face successfully!", "status": "success"}

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
        "http://localhost:8080/api/student-vectors",
        json={
            "username": username,
            "featureVector": array_vector,
        }
    )
    print("Status Code:", response.status_code)


    # gọi api student-vectors

    return {"message": "Register face successfully!", "status": "success"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
