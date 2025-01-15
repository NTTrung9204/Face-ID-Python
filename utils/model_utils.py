import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
import torch.nn as nn
from ultralytics import YOLO

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    return 1 - np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def preprocess_face(face_image: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # transform nhận vào một đối tượng thuộc kiểu PIL.Image.Image
    face_image = Image.fromarray(face_image)

    return transform(face_image).unsqueeze(0)


def extract_face_embedding(
    face_image: np.ndarray, extractor_model: nn.Module
) -> np.ndarray:
    face_tensor: torch.Tensor = preprocess_face(face_image)

    with torch.no_grad():
        embedding: torch.Tensor = extractor_model(face_tensor)

    return embedding[0].cpu().numpy()


def detect_faces(image: np.ndarray, detector_model: YOLO) -> np.ndarray:
    results = detector_model(image)
    faces: np.ndarray = results[0].boxes.xyxy.cpu().numpy()

    return faces


def extract_identity_embedding(
    extractor_model: nn.Module, folder_path: str = "identity"
) -> dict[str, list[np.ndarray]]:
    embeddings: dict[str, list[np.ndarray]] = {}

    for label in os.listdir(folder_path):
        label: str

        embeddings[label]: list[np.ndarray] = []  # type: ignore

        label_path: str = os.path.join(folder_path, label)

        for image_name in os.listdir(label_path):
            image_name: str
            image_path: str = os.path.join(label_path, image_name)

            embedding: np.ndarray = extract_face_embedding(
                cv2.imread(image_path), extractor_model
            )
            embeddings[label].append(embedding)

    return embeddings


def nearest_face(
    face_embedding: np.ndarray, identity_embedding: dict[str, list[np.ndarray]]
) -> tuple[str, float]:
    distances: dict[str, list[float]] = {}

    for label, embeddings in identity_embedding.items():
        label: str
        embeddings: list[np.ndarray]

        distances[label]: list[float] = []  # type: ignore

        for embedding in embeddings:
            embedding: np.ndarray

            # distance: float = np.linalg.norm(face_embedding - embedding)
            distance: float = cosine_similarity(face_embedding, embedding)

            distances[label].append(distance)

    min_distance: float = np.inf
    min_label: str = None

    for label, label_distances in distances.items():
        label: str
        label_distances: float

        if min(label_distances) < min_distance:
            min_distance: float = min(label_distances)
            min_label: str = label

    return min_label, min_distance


def identify_faces(
    faces: np.ndarray,
    identity_embedding: dict[str, list[np.ndarray]],
    image: np.ndarray,
    extractor_model: nn.Module,
) -> list[tuple[str, np.ndarray, float]]:
    identified_faces: list[tuple[str, np.ndarray, float]] = []

    for face in faces:
        face: np.ndarray

        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        face_image: np.ndarray = image[y1:y2, x1:x2]

        face_embedding: np.ndarray = extract_face_embedding(face_image, extractor_model)

        label, distance = nearest_face(face_embedding, identity_embedding)

        identified_faces.append((label, face, distance))

    return identified_faces
