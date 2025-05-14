import torchvision
#from torchvision import transforms

transformations = torchvision.transforms.Compose([
    
    # torchvision.transforms.toPILImage()
    
    torchvision.transforms.Resize(size=(244, 244)),
    
    torchvision.transforms.ToTensor(),
    
    torchvision.transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])