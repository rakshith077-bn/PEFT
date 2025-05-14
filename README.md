# Fine-Tuning Vision with Transformer for Image Classification

This version is under testing and is undergoing feature updates. To use this repo clone the previous stable version.

The current update supports Lion-Optimizer

## Overview
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. 
- I've currently fine-tuned the `google/vit-base-patch16-224` model to classify a custom image dataset efficiently, making it suitable for resource-constrained environments. 
 

## Getting Started
- Creates dependency issues for python3.11 version and above. Create a virtual environment with **python3.10**
```python
python3.10 -m venv venv
```
- Run 
```bash
chmod +x peft.sh
./peft.sh
```
### Note 
For macOS environment, tensorFlow-macos and tensorFlow-metal has to installed to enable GPU training. Recomended for faster training time if you have a larger dataset. 

## Usage
```python
python3 finetune.py --help # For finetuning your dataset
python3 feature_extraction.py --help # Extract features once you have finished fine tuning
```

- run PEFT on sample dataset.
```python
python3 finetune.py --data-path sample_data --num-epochs 10 --batch-size 16
```

- Extract features of the finetuned sample dataset
```python
python3 feature_extraction.py --model_path model.pth --dataset_path sample_data
```

### Dataset Format: 
The dataset is expected to be in the followving format:

```python
sample_data
├── folder 
    ├── class1
    ├── class2
    └── ...
```

## Extract Features
- Utilize the feature extraction method to extract embeddings. The output is a .CSV file. 
- Run the feature extraction with just a few arguments. 
```python
python feature_extraction.py --model_path <.pth path> --dataset_path <dataset_path>
```

## Model 
- **Model**: The `google/vit-base-patch16-224` Vision Transformer model is used, fine-tuned with LoRA.
- **Optimizer**: Lion-Optimizer
- **Loss Function**: **Cross Entropy Loss**. Calculated at the end of each epoch.

## Evaluation
After training, the model is evaluated on a test dataset, and the average test loss is reported. The current update uses stratified sampling.

## Output
- Outputs `.pth` file within the same directory. 

The current limitations are known. I'm pushing toward improving this repositry. If you identify an error, a possible fix or improvement, open a clearly defined pull request. 

## License
This project is open source and available under the [MIT License](LICENSE).
