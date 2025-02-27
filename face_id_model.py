from utils.model_utils import detect_faces, identify_faces, extract_identity_embedding
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch.nn as nn


class FaceIDModel:
    def __init__(self, detector_path: str, identity_folder: str = "identity") -> None:
        self.detector_model: YOLO = YOLO(detector_path, verbose=True)
        self.extractor_model: nn.Module = InceptionResnetV1(
            pretrained="vggface2"
        ).eval()

        self.identity_embedding: dict[str, list[np.ndarray]] = (
            extract_identity_embedding(self.extractor_model, identity_folder)
        )

    def query(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        faces: np.ndarray = detect_faces(frame, self.detector_model)
        identified_faces: list[tuple[str, np.ndarray, float]] = identify_faces(
            faces, self.identity_embedding, frame, self.extractor_model
        )

        return identified_faces
