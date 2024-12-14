# Fine-Tuning Vision Transformer for Image Classification

## Overview
This script allows CSCI-525.10 students to utilize a pre-trained PEFT model with their dataset by running a few simple commands.  
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification.
- I've currently fine-tuned the `google/vit-base-patch16-224` model to classify a custom image dataset efficiently, making it suitable for resource-constrained environments. Highly recomend GPU training for Advanced Phase 3.
- Save embeddings to a csv file, using the feature extraction method.  

## Getting Started
### Prerequisites

- Clone the project `git clone https://github.com/rakshith077-bn/mld-FineTune`
- make sure your currently in the same directory as `mld-FineTune`    

### It is recommended to use a virtual environment: 

```sh
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```
- Install the required packages
```sh
pip install -r requirements.txt
```

## Running PEFT on sample data
- run PEFT on a sample dataset. Check sample_data for the expected structure of the dataset


- Run the model with just a few arguments
```python
python finetune.py --data_path <dataset_path> --num_epochs 10 --batch_size 16
```

### Arguments
- `--data_path`: Path to the dataset folder.
- `--num_epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.

## Using Feature Extraction
- Utilize the feature extraction method to extract embeddings. The output is stored in a csv file following the norms defined in the CSCI-525.10 Machine Learning Design course.
- Run the feature extraction with just a few arguments. 
```python
python feature_extraction.py --model_path <.pth path> --dataset_path <dataset_path>
```

### Note 
- `--model_path <.pth path>` - This file contains the trained weights of the PEFT model. It is generated after the training phase and generally stored in the same directory. 

## Dataset Format: 
The dataset is expected to be in the following format. Do not worry about the split, it is done within the script

```python
sample_data
├── folder 
    ├── class1
    ├── class2
    └── ...
```

## PEFT
- **Model**: The `google/vit-base-patch16-224` Vision Transformer model is used, fine-tuned with LoRA applied to specific attention layers (`query` and `value` projections).
- You can experiment with different layers within `config`. Check out 'target_modules.txt'
- **Optimizer**: The **AdamW** optimizer is used with a learning rate of `1e-3`. You can change it to 1e-4 if your system can take it.
- **Loss Function**: **Cross Entropy Loss** is used for classification. You see this score at the end of each epoch.
- **Device**: The script automatically detects and uses a GPU if available; otherwise, it falls back to the CPU.
- **Note**: It is highly recommended that you are on GPU, since you'll be loading your entire dataset. If else, experiment. This was developed on macOS, and I had tensorFlow-metal and tensorFlow-macos installed. Shouldn't affect your work but if it does, double check this aspect. 

## Evaluating the Model
After training, the model is evaluated on a test dataset, and the average test loss is reported:

## Saving the Model
- Outputs .pth file which can be used for feature extraction with a deep learner of your choice.

## Citation
If you use this repository in your work, provide the following citation:

```bibtex
@misc{rakshith2024customPEFTscript,
  author = {Rakshith B N},
  title = {Parameter-Efficient Fine-Tuning of Vision Transformer for Image Classification for CSCI-525.10 Machine Learning Design},
  year = {2024},
  published = {\url{https://github.com/rakshith077-bn/MLD-Fintune}},
}
```

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- The Vision Transformer model used in this project was pre-trained on the **ImageNet-21k** dataset. https://huggingface.co/models?search=google/vit. Neccessary citations have been provided in places used. 
- This can be used for CSCI-525.10 as a part of Phase 3 option 1 (PEFT)
