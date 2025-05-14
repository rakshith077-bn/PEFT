from transformers import ViTForImageClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def prepare_model(device):
    model = ViTForImageClassification.from_pretrainned('google/vit-base-patch16-224', torch_dtype=torch.float16, device_map='auto')

    model = model.to(device)
    config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.1,
        target_modules = ['encoder.layer.11.attention.attention.query', 'encoder.layer.11.attention.attention.value'],
        inference_mode = False
    )

    return get_peft_model(model, config)
