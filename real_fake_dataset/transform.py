import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

data_transforms = transforms.Compose(
    [
        transforms.Lambda(
            lambda img: TF.crop(
                img, top=3, left=3, height=img.size[1] - 6, width=img.size[0] - 6
            )
        ),
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
