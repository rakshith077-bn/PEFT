# ┌────────────────────────────────────────────────────────────────────────────── ┐
# │ Citation                                                                      │
# ├────────────────────────────────────────────────────────────────────────────── ┤
# │ @misc{rakshith2024customPEFTscript,                                           │
# │   author = {Rakshith B N},                                                    │
# │   title = {Parameter-Efficient Fine-Tuning of Vision Transformer for Image    │
# │            Classification for CSCI-525.10 Machine Learning Design},           │
# │   year = {2024},                                                              │
# │   howpublished = {\url{https://github.com/rakshith077-bn/MLD-Fintune}},       │
# │                                                                               │
# │ }                                                                             │
# └────────────────────────────────────────────────────────────────────────────── ┘
#
# If you use this code in your work, cite it using reference.


import argparse
import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from load_dataset import ImageDataset
from utilities import transformations
import os
import torchvision


# experiment with different preprocessing approaches if you wish
 
transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def main():
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Example')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs. Minimum is 10')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Define batch size. Recomended 16, if you are on a slower computer try 8')
    
    #parser.add_argument('--num_classes', type=int, help='Number of classes present in your dataset. Count from 1 not 0')
    
    args = parser.parse_args()

    # Load dataset using load_dataset.py
    dataset_path = args.data_path
    
    BATCH = args.batch_size
    
    dataset = ImageDataset(root_dir=dataset_path, transform=transformations)
    
    loader = torch.utils.data.DataLoader(
        dataset_path,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
    )
    
    # print("Loaded Dataset")

    # Split dataset into train and test
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    """
    
    @misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    
    @inproceedings{deng2009imagenet,
    title={Imagenet: A large-scale hierarchical image database},
    author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
    booktitle={2009 IEEE conference on computer vision and pattern recognition},
    pages={248--255},
    year={2009},
    organization={Ieee}
    }
    
    """

    # using patch16-224 since I'm quite familiar with the target modules
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torch_dtype=torch.float16, device_map='auto')

    # Apply parameter-efficient fine-tuning by LORA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['encoder.layer.11.attention.attention.query', 'encoder.layer.11.attention.attention.value'], # feel free to experiment with different layers. Check target_modules.txt
        inference_mode=False # run inferencing if you wish
    )
    
    model = get_peft_model(model, config)

    # Recomended for multiple runs under the same .venv 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Know your device. If it is on CPU it is likely take a long time. 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using Device: {device}")

    # Define optimizer and loss function
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # change your learning rate if your system can take it. Consider other factors as well. 
    
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = args.num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('Training Loss: {avg_train_loss:.4f}')

    # Testing your loss score
    model.eval()
    test_loss = 0
    
    
    with torch.no_grad():
    
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(pixel_values=images).logits
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    print(f'Loss: {avg_test_loss:.4f}')

    # Save the model
    model_save_path = 'model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved Model: {model_save_path}')

if __name__ == '__main__':
    main()
