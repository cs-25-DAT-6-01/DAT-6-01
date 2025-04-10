import time

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "6"

# Path to the trained model/tokenizer
base_model_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"
model_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2-fine_tuning-1"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2-fine_tuning-1"

# Load the model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
#model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model)
print("Memory used (MBs):", model.get_memory_footprint() / 1e6)
model.eval()

# Input text
input_text = "What is New York City?"
print("Input text:", input_text)

# Start time
start_time = time.time()

# Tokenize the input text
inputs = tokenizer.encode_plus(
    input_text,
    return_tensors="pt",
    #padding="max_length",
    #truncation=True,
    #max_length=150,
    #pad_to_max_length=True,
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate the output (prediction)
# https://huggingface.co/docs/transformers//generation_strategies#generation-strategies
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=250,
    #temperature=0.7,
    #top_k=50,
    #top_p=0.9,
    repetition_penalty=1.3,
    #do_sample=True,
)
print(output)
# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# End time
end_time = time.time()
inference_time = end_time - start_time

# Print the generated text and time taken
print("Text generated:", generated_text)
print("Inference time (seconds):", inference_time)
