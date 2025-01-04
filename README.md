# Fine-Tuning Vision Transformer for Image Classification

## Overview
This script allows users to utilize a pre-trained PEFT model with their image dataset by running a few simple commands. Note this script intendes use for only image datasets.  
- This was developed with an aim to demonstrate parameter-efficient fine-tuning of a Vision Transformer (ViT) model using **LoRA (Low-Rank Adaptation)** for image classification. The model was tested and configured for the `Chula-ParasiteEgg11` dataset - https://icip2022challenge.piclab.ai/dataset/.
- I've currently fine-tuned the `google/vit-base-patch16-224` model to classify a custom image dataset efficiently, making it suitable for resource-constrained environments. The script utilizes your GPUs and works fine for both Windows and macOS.
- Save the fine tuned weights to a csv file, using the `feature extraction` method.  
 

## Getting Started
- Run `python3 finetune.py --help`

### Prerequisites
- Clone the project `git clone https://github.com/rakshith077-bn/mld-FineTune`

## Note
- The `finetune.py` script uses **torch** which is not compatible with **python@3.12** and above. To resolve this follow the excat command whilst creating your virtual environment.
```sh
python3.10 -m venv venv # To prevent dependency issues with torch
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```
- You are just creating a python3.10 virutal environment. This should not create any dependency issues.

- Install the required packages
```sh
pip install -r requirements.txt
```
**Note**: If your using a macOS environment, the requirements.txt file does not have tensorFlow-macos and tensorFlow-metal, you must install it separately.

## Running PEFT on sample data
- run PEFT on the given sample dataset. Check sample_data for the expected structure of your dataset.

### Dataset Format: 
The dataset is expected to be in the following format. The scrip uses stratified sampling to split your dataset.

```python
sample_data
├── folder 
    ├── class1
    ├── class2
    └── ...
```


## Run the model with just a few arguments
```python
python finetune.py --data_path <dataset_path> --num_epochs 10 --batch_size 16
```

### Arguments
- `--data_path`: Path to the dataset folder.
- `--num_epochs`: Number of epochs.
- `--batch_size`: Batch size.

## Extract Features
- Utilize the feature extraction method to extract embeddings. The output is stored in a csv file. 
- Run the feature extraction with just a few arguments. 
```python
python feature_extraction.py --model_path <.pth path> --dataset_path <dataset_path>
```

## PEFT
- **Model**: The `google/vit-base-patch16-224` Vision Transformer model is used, fine-tuned with LoRA applied to specific attention layers (`query` and `value` projections).
- You can experiment with different layers within `config`. Check 'target_modules.txt'
- **Optimizer**: The **AdamW** optimizer is used with a learning rate of `1e-3`. 
- **Loss Function**: **Cross Entropy Loss** is used for classification. You see this score at the end of each epoch.
- **Device**: The script automatically detects and uses a GPU if available; otherwise, it falls back to the CPU.
- **Note**: It is highly recommended that you are on GPU, since you'll be loading your entire dataset. 

## Model Evaluation
After training, the model is evaluated on a test dataset, and the average test loss is reported. The current update uses stratified sampling to prevent dataset contamination.

## Model Output
- Outputs `.pth` file within the same directory after training completion, which can be used for feature extraction with a deep learner of your choice.

```
The current limitations are known. I'm pushing toward improving this repositry. If you identify an error, a possible fix or sugestion, open a clearly defined pull request. 
```

## Citation
If you use this repository in your work, provide the following citation:

```bibtex
@misc{rakshith2024customPEFTscript,
  author = {Rakshith B N},
  title = {Parameter-Efficient Fine-Tuning of Vision Transformer for Image Classification },
  year = {2024},
  published = {\url{https://github.com/rakshith077-bn/MLD-Fintune}},
}
```

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- The Vision Transformer model used in this project was pre-trained on the **ImageNet-21k** dataset. https://huggingface.co/models?search=google/vit. Neccessary citations have been provided in places used. 
