import time

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rouge_score import rouge_scorer

# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "10"
alpha = "0.5"
beta = "0.5"

# Path to the trained model/tokenizer
# = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"
model_path = f"model-{model_name}_epochs-{amount_of_epochs}_webquestions_alpha-{alpha}_beta-{beta}"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}_webquestions_alpha-{alpha}_beta-{beta}"

# Load the model and tokenizer
#base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
#model = PeftModel.from_pretrained(base_model, model_path)
#model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model)
print("Memory used (MBs):", model.get_memory_footprint() / 1e6)
model.eval()

# Input text
input_text = "what is the name of justin bieber brother?" # https://huggingface.co/datasets/Stanford/web_questions
print("Input text:", input_text)

# Tokenize the input text
inputs = tokenizer.encode_plus(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
)

input_ids = inputs["input_ids"].to(device)

# Start time
start_time = time.time()

# Generate the output (prediction)
# https://huggingface.co/docs/transformers//generation_strategies#generation-strategies
output = model.generate(
    input_ids,
    use_cache=False,
    kv_cache=None,
)
torch.cuda.synchronize()  # Synchronize CUDA to ensure all operations are complete before measuring time
# End time
end_time = time.time()
inference_time = end_time - start_time

print(output)
# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

reference_text = ("Jazmyn Bieber"
                  "Jaxon Bieber") # This will need to be updated.

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
scores = scorer.score(reference_text, generated_text)

# Print the generated text and time taken
print("Text generated:", generated_text)
print("Inference time (seconds):", inference_time)
print("ROUGE-1:", scores['rouge1'])
print("ROUGE-2:", scores['rouge2'])
print("ROUGE-L:", scores['rougeL'])
print("ROUGE-Lsum:", scores['rougeLsum'])
