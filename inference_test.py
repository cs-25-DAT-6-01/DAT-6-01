import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "6"

# Path to the trained model/tokenizer
model_path = f"model-{model_name}_epochs-{amount_of_epochs}"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input text
input_text = "How many people live in New York City?"

# Start time
start_time = time.time()

# Tokenize the input text
inputs = tokenizer.encode_plus(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=150,
    pad_to_max_length=True,
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate the output (prediction)
# Max length is the maximum number of tokens to generate
output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# End time
end_time = time.time()
inference_time = end_time - start_time

# Print the generated text and time taken
print("Text generated:", generated_text)
print("Inference time:", inference_time)
