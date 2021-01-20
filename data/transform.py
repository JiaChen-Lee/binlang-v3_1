# Created by Jiachen Li at 2021/1/18 21:34
from torchvision import transforms
from cvtorchvision import cvtransforms


def create_transform(load_img_with, resized_size):
    if load_img_with == "OpenCV":
        transform = cvtransforms.Compose([
            cvtransforms.CenterCrop(1024),
            cvtransforms.Resize((resized_size, resized_size)),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(1024),
            transforms.Resize((resized_size, resized_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform
