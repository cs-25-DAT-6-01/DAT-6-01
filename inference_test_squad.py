import time
import string
import torch

from statistics import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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

# Modle name and parameters
MODEL_NAME = "openai-community-gpt2"
teacher_name = "openai-community/gpt2"
EPOCHS = 10
ALPHA, LAMBDA, BETA, GAMMA, TEMP = 8, 0.7, 0.3, 1.5, 2

# Define the model path based on the parameters
MODEL_PATH = (
    f"model-{MODEL_NAME}_epochs-{EPOCHS}_squad_"
    f"alpha-{ALPHA}_beta-{BETA}_lambd-{LAMBDA}_gamma-{GAMMA}_temperature-{TEMP}"
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# Load the SQuAD dataset
test_dataset = load_dataset("squad", split="validation[:1000]")

# Initialize lists to store inference times and results
inference_times = []
results = []

# Prompt the model to generate questions based on the context and answer
for example in test_dataset:
    prompt = f"Context: {example['context']}\nAnswer: {example['answers']['text']}\nQuestion:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Start time and generate the question
    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
    )
    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    inference_times.append(end - start)
    # Decode the output and extract the generated question
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Question:" in generated_text:
        # Extract the question part if "Question:" is present
        generated_question = generated_text.split("Question:")[-1].strip()
    else:
        # If "Question:" is not present, use the entire generated text
        generated_question = generated_text.strip()
    # Append the generated question to results
    results.append(generated_question)

# Initialize lists for exact matches and F1 scores
exact_matches = []
f1_scores = []

# Compute exact match and F1 scores for each generated question
for example, pred in zip(test_dataset, results):
    true_answer = example["question"] if example["question"] else ""
    exact_matches.append(compute_exact(pred, true_answer))
    f1_scores.append(compute_f1(pred, true_answer))

# Print the average inference time, exact match, and F1 scores
print("Average inference latency :", mean(inference_times))
print("Exact-Match               :", mean(exact_matches))
print("Token-level F1            :", mean(f1_scores))
