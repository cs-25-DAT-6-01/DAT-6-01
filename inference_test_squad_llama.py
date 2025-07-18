import time
import string
from statistics import mean

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load

# Normalize the answer by removing articles, punctuation, and extra whitespace
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return ' '.join([w for w in text.split() if w.lower() not in ('a', 'an', 'the')])
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Compute exact match
def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))

# Function to compute F1 score
def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(true_tokens)
    return 2 * prec * rec / (prec + rec)

# BitsAndBytesConfig for quantization
bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

# Model name and parameters
MODEL_NAME = "meta-llama-Llama-3.2-1B"
teacher_name = "meta-llama/Llama-3.2-1B"
EPOCHS = 10
ALPHA, LAMBDA, BETA, GAMMA, TEMP = 8, 0.7, 0.3, 1.5, 2

# Define the model path based on the parameters
MODEL_PATH = (
    f"model-{MODEL_NAME}_epochs-{EPOCHS}_squad_"
    f"alpha-{ALPHA}_beta-{BETA}_lambd-{LAMBDA}_gamma-{GAMMA}_temperature-{TEMP}"
)

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, device_map="auto", torch_dtype="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# Load the SQuAD dataset
test_dataset = load_dataset("squad", split="validation[:1000]")

inference_times = []
results = []

# Preprocess the dataset to create prompts
for example in test_dataset:
    # Create a prompt for the model
    prompt = f"Context: {example['context']}\nAnswer: {example['answers']['text']}\nQuestion:"
    # Input the prompt into the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Start time and generate the output
    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    inference_times.append(end - start)
    # Decode the output and extract the generated question
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Split "question" from the generated text
    if "Question:" in generated_text:
        generated_question = generated_text.split("Question:")[-1].strip()
    else:
        generated_question = generated_text.strip()
    # Append the generated question to results
    results.append(generated_question)

# Initialize lists to store exact matches and F1 scores
exact_matches = []
f1_scores = []

# Compute exact matches and F1 scores for each example
for example, pred in zip(test_dataset, results):
    true_answer = example["question"] if example["question"] else ""
    exact_matches.append(compute_exact(pred, true_answer))
    f1_scores.append(compute_f1(pred, true_answer))

# Print the average inference latency, exact match, and F1 scores
print("Average inference latency :", mean(inference_times))
print("Exact-Match               :", mean(exact_matches))
print("Token-level F1            :", mean(f1_scores))
