# Fine-Tuning Vision with Transformer for Image Classification

The current update supports Lion-Optimizer

## Overview
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. 
- I've currently fine-tuned the `google/vit-base-patch16-224` model to classify a custom image dataset efficiently, making it suitable for resource-constrained environments. 
 

## Getting Started
- `python3 finetune.py --help`
- `python3 feature_extraction.py --help`

## Note
- `finetune.py` uses **torch** which creates dependency issues with **python@3.12** and higher versions. To prevent this, create your venv with `python3.10`

**Note**: For macOS environment, tensorFlow-macos and tensorFlow-metal has to installed to enable GPU training. Recomended for faster training time if you have a larger dataset. 

## Running PEFT on sample data
- run PEFT on the given sample dataset. 

```python
python3 finetune.py --data_path sample_data --num_epochs 10 --batch_size 16
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

## Citation
If you use this repository in your work, provide the following citation:

```bibtex
@misc{rakshith2024PEFTscript,
  author = {Rakshith B N},
  title = {Parameter-Efficient Fine-Tuning of Vision Transformer for Image Classification },
  year = {2024},
  published = {\url{https://github.com/rakshith077-bn/PEFT}},
}
```

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- The Vision Transformer model used in this project was pre-trained on the **ImageNet-21k** dataset. https://huggingface.co/models?search=google/vit.  
- 
