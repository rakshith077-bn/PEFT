# Fine-Tuning Vision Transformer for Image Classification

## Overview
This script allows users to utilize a pre-trained PEFT model with their image dataset by running a few simple commands. Note this script intendes use for only image datasets.  
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. The model was tested and configured for the `Chula-ParasiteEgg11` dataset - https://icip2022challenge.piclab.ai/dataset/.
- I've currently fine-tuned the `google/vit-base-patch16-224` model to classify a custom image dataset efficiently, making it suitable for resource-constrained environments. 
 

## Getting Started
- Run `python3 finetune.py --help`
- Run `python3 feature_extraction.py --help`

## Note
- The `finetune.py` script uses **torch** which is not compatible with **python@3.12** and higher versions. To prevent dependency issues create your venv with `python3.10`

```sh
python3.10 -m venv venv 
source venv/bin/activate  # macOS/Linux
```

- Install the required packages
```sh
pip install -r requirements.txt
```
**Note**: If your using a macOS environment, the requirements.txt file does not have tensorFlow-macos and tensorFlow-metal, you must install it separately for GPU enabled training.

## Running PEFT on sample data
- run PEFT on the given sample dataset. 

### Dataset Format: 
The dataset is expected to be in the following format. 

```python
sample_data
├── folder 
    ├── class1
    ├── class2
    └── ...
```

## Fine tune your model with just a few commands
```python
python3 finetune.py --data_path <dataset_path> --num_epochs 10 --batch_size 16
```

## Extract Features
- Utilize the feature extraction method to extract embeddings. The output is a .CSV file. 
- Run the feature extraction with just a few arguments. 
```python
python feature_extraction.py --model_path <.pth path> --dataset_path <dataset_path>
```

## Model 
- **Model**: The `google/vit-base-patch16-224` Vision Transformer model is used, fine-tuned with LoRA applied to specific attention layers (`query` and `value` projections).
- You can experiment with different layers within `config`. Check 'target_modules.txt'
- **Optimizer**: The **AdamW** optimizer is used. 
- **Loss Function**: **Cross Entropy Loss** is employed. You see this score at the end of each epoch.
- **Device**: The script automatically detects and uses a GPU if available.

## Evaluation
After training, the model is evaluated on a test dataset, and the average test loss is reported. The current update uses stratified sampling to prevent dataset contamination.

## Output
- Outputs `.pth` file within the same directory. 

The current limitations are known. I'm pushing toward improving this repositry. If you identify an error, a possible fix or sugestion, open a clearly defined pull request. 

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
