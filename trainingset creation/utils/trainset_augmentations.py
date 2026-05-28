import albumentations as A

image_transform = A.Compose([
    A.SmallestMaxSize(512), 
    A.CenterCrop(512, 512),
])

strong_transform = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.2),
    A.RandomResizedCrop(size = (512,512), scale=(0.08, 1.0), ratio=(0.75, 1.0/0.75), p=0.2),

    A.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.05, p=0.8),
    A.ToGray(p=0.2),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
    A.ImageCompression('jpeg', (70,95), p=0.2),
    A.GaussianBlur(blur_limit=(3,5), p=0.1),
    A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(96, 96), hole_width_range=(96, 96), fill=0, p=0.2),
])