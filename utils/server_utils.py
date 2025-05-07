from werkzeug.datastructures import FileStorage
import os
import mysql.connector
import json
import numpy as np
import torch
from datetime import datetime

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
        
    def get_current_lesson_for_student(student_id):
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='auto_attendance_pbl5',
            port=3307,
        )

        try:
            cursor = connection.cursor(dictionary=True)
            
            # Get current time and date
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%Y-%m-%d")
            
            # print(f"Current date: {current_date}, Current time: {current_time}")
            
            # Find active lessons (current time between start_time and end_time)
            sql = """
            SELECT id, start_time, end_time, lesson_date FROM lesson 
            WHERE lesson_date = %s 
            AND TIME(%s) BETWEEN TIME(start_time) AND TIME(end_time) 
            AND is_completed = 0
            """
            
            cursor.execute(sql, (current_date, current_time))
            active_lessons = cursor.fetchall()
            
            # print(f"Active lessons: {active_lessons}")
            
            if not active_lessons:
                # For testing: Try to find any lesson today regardless of time
                sql = """
                SELECT id, start_time, end_time, lesson_date FROM lesson 
                WHERE lesson_date = %s 
                AND is_completed = 0
                """
                cursor.execute(sql, (current_date,))
                today_lessons = cursor.fetchall()
                # print(f"Today's lessons: {today_lessons}")
                
                # Check if there's a lesson today, but it hasn't started or has ended
                if today_lessons:
                    for lesson in today_lessons:
                        print(f"Checking lesson {lesson['id']}: start={lesson['start_time']}, end={lesson['end_time']}")
                
                return None
                
            # Check if student is registered for any active lesson
            for lesson in active_lessons:
                print(f"Checking attendance for lesson {lesson['id']}")
                # Check attendance_check records
                sql = """
                SELECT * FROM attendance_check 
                WHERE lesson_id = %s AND student_id = %s
                """
                cursor.execute(sql, (lesson['id'], student_id))
                attendance_record = cursor.fetchone()
                
                if attendance_record:
                    # print(f"Found attendance record for student {student_id} in lesson {lesson['id']}")
                    return lesson['id']
            
            # print(f"No attendance records found for student {student_id} in any active lessons")
            return None

        finally:
            cursor.close()
            connection.close()

    def get_student_by_id(student_id):
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='auto_attendance_pbl5',
            port=3307,
        )

        try:
            cursor = connection.cursor(dictionary=True)
            
            sql = "SELECT * FROM student WHERE id = %s"
            cursor.execute(sql, (student_id,))
            student = cursor.fetchone()
            
            return student

        finally:
            cursor.close()
            connection.close()
        