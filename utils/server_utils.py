from werkzeug.datastructures import FileStorage
import os

class ServerUtils:
    def __init__(self) -> None:
        pass

    def save_image(label: str, image: FileStorage) -> None:
        directory = f"identity/{label}"
        os.makedirs(directory, exist_ok=True)

        image.save(f"{directory}/{image.filename}")
