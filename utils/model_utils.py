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

def euclid_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.linalg.norm(A - B)

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

def detect_phone(image: np.ndarray, detector_model: YOLO) -> np.ndarray:
    results = detector_model(image)
    phones: np.ndarray = results[0].boxes.xyxy.cpu().numpy()

    return phones


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

    # print("LEN EMB", len(identity_embedding.items()))

    for label, embeddings in identity_embedding.items():
        label: str
        embeddings: list[np.ndarray]

        distances[label]: list[float] = []  # type: ignore

        for embedding in embeddings:
            embedding: np.ndarray

            distance: float = cosine_similarity(face_embedding, embedding)

            # print(f"Distance between {label} and face: {distance:.4f}")

            distances[label].append(distance)

    min_distance: float = np.inf
    min_label: str = None

    for label, label_distances in distances.items():
        label: str
        label_distances: float

        # print(f"Distances for {label}: {label_distances}")

        if min(label_distances) < min_distance:
            min_distance: float = min(label_distances)
            min_label: str = label

    # print(distances)

    return min_label, min_distance

def predict_real_fake(
    face_image: np.ndarray, classifier_model: torch.nn.Module, device: torch.device
):
    # Convert BGR to RGB for PIL (if face_image is in BGR)
    # face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Now create PIL image from RGB array
    image = Image.fromarray(face_image)

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(400),  # Resize to a larger size first (zoom in)
    #         transforms.CenterCrop(299),  # Then crop the center to 299x299
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    classifier_model.eval()
    classifier_model.to(device)
    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.cpu().numpy()
def identify_faces(
    faces: np.ndarray,
    phones: np.ndarray,
    identity_embedding: dict[str, list[np.ndarray]],
    image: np.ndarray,
    extractor_model: nn.Module,
    real_fake_model: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> list[tuple[str, np.ndarray, float, int, np.ndarray]]:
    identified_faces = []

    for face in faces:
        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Check if face is inside any phone bounding box
        is_inside_phone = False
        for phone in phones:
            px1, py1, px2, py2 = map(int, phone)
            if (x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2):
                is_inside_phone = True
                break

        face_image = image[y1:y2, x1:x2]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_embedding = extract_face_embedding(face_image, extractor_model)
        label, distance = nearest_face(face_embedding, identity_embedding)
        
        # If face is inside phone or model predicts fake, mark as fake
        real_fake_class, real_fake_probs = predict_real_fake(face_image, real_fake_model, device)
        if is_inside_phone or real_fake_class == 1:
            label = "Fake"
            real_fake_class = 1
        
        identified_faces.append((label, face, distance, real_fake_class, real_fake_probs[0]))

    return identified_faces

def numpy_array_to_list(identity_embedding):
    return {key: [arr.tolist() for arr in value] for key, value in identity_embedding.items()}
