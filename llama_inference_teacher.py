import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from rouge_score import rouge_scorer
from huggingface_hub import login


login(os.getenv("HF_TOKEN"))
# Define file name and such
model_name = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = list(model.hf_device_map.values())[0]

# Set the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

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
    #padding="max_length",
    #truncation=True,
    #max_length=150,
    #pad_to_max_length=True,
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Start time
start_time = time.time()

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
print(output)
# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# End time
end_time = time.time()
inference_time = end_time - start_time

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
