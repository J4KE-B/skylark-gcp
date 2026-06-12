import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_KP = A.KeypointParams(format="xy", remove_invisible=False)


def get_train_transform(width: int, height: int):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.8, border_mode=0, fill=0),
        A.RandomScale(scale_limit=0.15, p=0.5),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, fill=0),
        A.CenterCrop(height=height, width=width),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ], keypoint_params=_KP)


def get_val_transform(width: int, height: int):
    return A.Compose([
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ], keypoint_params=_KP)
