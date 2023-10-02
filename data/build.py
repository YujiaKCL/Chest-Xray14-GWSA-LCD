from torchvision import transforms


def build_transform(train, args):
    if train:
        transform = transforms.Compose((
            transforms.RandomResizedCrop(int(args.img_size / 0.875),
            scale = (0.8, 1.0),
            ),
            transforms.RandomRotation(7),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ))
    else:
        transform = transforms.Compose((
            transforms.Resize(int(args.img_size / 0.875)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ))
    return transform
