from utils.model_utils import detect_faces, identify_faces, extract_identity_embedding, detect_phone
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch.nn as nn
from real_fake_dataset.facenet import create_model
import torch

class FaceIDModel:
    def __init__(self, detector_path: str, identity_folder: str = "identity") -> None:
        self.detector_model: YOLO = YOLO(detector_path, verbose=False)
        self.phone_detector_model: YOLO = YOLO("model/phone.pt", verbose=False)
        self.extractor_model: nn.Module = InceptionResnetV1(
            pretrained="vggface2"
        ).eval()
        self.classifier_model: nn.Module = create_model(num_classes=2)
        self.classifier_model.load_state_dict(torch.load("model/real_fake_model_v1.pth"))
        self.classifier_model.eval()

        self.identity_embedding: dict[str, list[np.ndarray]] = (
            extract_identity_embedding(self.extractor_model, identity_folder)
        )

    def query(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        faces: np.ndarray = detect_faces(frame, self.detector_model)
        phones: np.ndarray = detect_phone(frame, self.phone_detector_model)
        identified_faces: list[tuple[str, np.ndarray, float]] = identify_faces(
            faces, phones, self.identity_embedding, frame, self.extractor_model, self.classifier_model
        )

        return identified_faces
    
    def set_identity_embedding(self, identity_embedding: dict[str, list[np.ndarray]]) -> None:
        self.identity_embedding = identity_embedding
