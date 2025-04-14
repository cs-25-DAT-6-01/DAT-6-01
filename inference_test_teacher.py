import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login


login(os.getenv("HF_TOKEN"))
# Define file name and such
model_name = "openai-community/gpt2-large"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input text
input_text = "What is New York City?"

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
    max_new_tokens=100,
    #temperature=0.7,
    #top_k=50,
    #top_p=0.9,
    repetition_penalty=1.3,
    #do_sample=True,
    use_cache=False,
    kv_cache=None,
)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# End time
end_time = time.time()
inference_time = end_time - start_time

# Print the generated text and time taken
print("Text generated:", generated_text)
print("Inference time (seconds):", inference_time)
