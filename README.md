# Fine-Tuning Vision with Transformer for Image Classification

The current update supports Lion-Optimizer

## Overview
- PEFT was developed to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. 

## Getting Started
- Creates dependency issues with when venv is on python3.11 version and more. Instead create venv with **python3.10** 

```python
python3.10 -m venv venv
```

- Clone repo and run 
```bash
chmod +x peft.sh
./peft.sh
```

### Note 
For macOS environment, tensorFlow-macos and tensorFlow-metal is recomended.

## Usage
```
cd src 
```

```python
python3 finetune.py --help 
python3 feature_extraction.py --help 
```

- run PEFT on the given sample dataset
```python
python3 finetune.py --data-path sample_data --num-epochs 10 --batch-size 16
```

- Extract features with
```python
python3 feature_extraction.py --model_path model.pth --dataset_path sample_data
```

### Dataset Format: 
The dataset is expected to be in the following format:

```python
sample_data
├── class 1
    ├── img1
    ├── img2
    └── ...
```

## Evaluation
The model is evaluated on a test dataset, and the average test loss is reported at the end of training. Train loss is reported at the end of each epoch.

## Output
- Outputs `.pth` file within the same directory. 

The current limitations are known. I'm pushing toward improving this repositry. If you identify an error, a possible fix or improvement, open a clearly defined pull request.

## License
This project is open source and available under the [MIT License](LICENSE).
