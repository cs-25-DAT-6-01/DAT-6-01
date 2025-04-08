import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "6"

# Path to the trained model/tokenizer
model_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model.get_memory_footprint()/1e6)
print(model)

#model = prepare_model_for_kbit_training(model)

#config = LoraConfig(
#    r = 16,
#    lora_alpha = 32,
#    bias =  "none",
#    task_type = "CAUSAL_LM",
#    target_modules=[]
#)