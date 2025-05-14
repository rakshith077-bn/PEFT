# Fine-Tuning Vision with Transformer for Image Classification

The current update supports Lion-Optimizer and is best suited for a medium sized image datasets.

## Overview
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. 
- Best suited for resource constrained environments that wants to train small to medium sized image datasets.
 

## Getting Started
- Known to create dependency issues with python3.11 version and more. Create venv with **python3.10** and follow the given steps
```python
python3.10 -m venv venv
source venv/bin/activate
git clone https://github.com/rakshith077-bn/PEFT/tree/test
```
- Run 
```bash
chmod +x peft.sh
./peft.sh
```
### Note 
For macOS environment, tensorFlow-macos and tensorFlow-metal has to installed to enable GPU training. Recomended for faster training time if you have a larger dataset. 

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
├── folder 
    ├── class1
    ├── class2
    └── ...
```

## Extract Feature Embeddings
```python
python feature_extraction.py --model_path <.pth path> --dataset_path <dataset_path>
```

## Evaluation
The model is evaluated on a test dataset, and the average test loss is reported. The current update uses stratified sampling.

## Output
- Outputs `.pth` file within the same directory. 

The current limitations are known. I'm pushing toward improving this repositry. If you identify an error, a possible fix or improvement, open a clearly defined pull request.

## License
This project is open source and available under the [MIT License](LICENSE).
