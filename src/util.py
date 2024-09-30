def prepare_data(J): # J is the size of the image (J x J)
    from torchvision import datasets, transforms
    return datasets.MNIST(
            root='~/Documents/Inverse-heat/data',
            download=True, train=False,
            transform=transforms.Compose([
                transforms.Resize((J, J)),
                transforms.ToTensor()
            ]))