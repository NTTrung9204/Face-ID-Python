from werkzeug.datastructures import FileStorage
import os
import mysql.connector
import json
import numpy as np
import torch

from utils.model_utils import extract_face_embedding, nearest_face

class ServerUtils:
    def __init__(self) -> None:
        pass

    def save_image(label: str, image: FileStorage) -> None:
        directory = f"identity/{label}"
        os.makedirs(directory, exist_ok=True)

        image.save(f"{directory}/{image.filename}")
        
    def fetch_student_vectors() -> list[np.ndarray]:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='auto_attendance_pbl5',
            port=3307,
        )

        try:
            cursor = connection.cursor(dictionary=True)
            sql = "SELECT * FROM student_vector"
            cursor.execute(sql)
            results = cursor.fetchall()

            vectors = {}
            for row in results:
                student_id = row['student_id']
                feature_vector = np.array(json.loads(row['feature_vector']))
                # vectors.append(feature_vector)
                vectors[student_id] = feature_vector

            return vectors

        finally:
            cursor.close()
            connection.close()

    def identify_student(student_vectors, face_image, extractor_model):
        face_embedding: np.ndarray = extract_face_embedding(face_image, extractor_model)

        label, distance = nearest_face(face_embedding, student_vectors)
        
        return label, distance
        