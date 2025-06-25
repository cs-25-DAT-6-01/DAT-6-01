import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rouge_score import rouge_scorer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utility import perplexity_for_llama
from utility import filter_lines
from collections import defaultdict

# Define file name and such
llama_model_name = "meta-llama-Llama-3.2-1B"
amount_of_epochs = "10"
alpha = "6"
lambd = "0.5"
beta = "0.5"
gamma = "1"
temperature = "3"

# Path to the trained model/tokenizer
model_path = f"model-{llama_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"
tokenizer_path = f"model-{llama_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the wikitext dataset
print("Loading wikitext dataset")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Filter and preprocess the dataset
test_dataset = test_dataset.map(lambda example: {'text': filter_lines(example['text'])})
test_dataset = test_dataset.filter(lambda example: len(example['text']) > 0)
test_dataset = test_dataset.select(range(1000))

# Tokenize the dataset
def tokenize_function(examples):
    return {
        "input_ids": tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128).input_ids,
        "attention_mask": tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128).attention_mask,
        "text": examples['text']
    }

# Tokenize the test dataset
print("Starting tokenization")
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'])

# Print model details
print(model)
print("Memory used (MBs):", model.get_memory_footprint() / 1e6)
model.eval()

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
total_inference_time = 0
total_score = defaultdict(list)

# Get the first device from the model's device map
first_device = list(model.hf_device_map.values())[0]

# Generate outputs and compute scores
for i in range(len(test_dataset)):
    input_ids = test_dataset[i]["input_ids"].unsqueeze(0).to(first_device)
    attention_mask = test_dataset[i]["attention_mask"].unsqueeze(0).to(first_device)
    # Get the reference text
    reference_text = test_dataset[i]["text"]
    # Decode the input_ids to get the input text
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Start time
    start_time = time.time()
    # Generate output using the model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        kv_cache=None,
    )
    torch.cuda.synchronize()  # Synchronize CUDA to ensure all operations are complete before measuring time
    # End time
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    # Decode the output and extract the generated text
    new_output = tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, "")
    # Score the generated output against the reference text
    score = scorer.score(reference_text, new_output)
    for metric, score_values in score.items():
        total_score[metric].append(score_values)

# Print average inference time
average_inference_time = total_inference_time / len(test_dataset)
print("Average inference time (seconds):", average_inference_time)

# Calculate average scores for each metric
average_scores = {}
for metric, scores in total_score.items():
    precisions = [score.precision for score in scores]
    recalls = [score.recall for score in scores]
    fmeasures = [score.fmeasure for score in scores]

    average_scores[metric] = {
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'fmeasure': sum(fmeasures) / len(fmeasures),
    }
    
# Print average scores
for metric, avg_score in average_scores.items():
    print(f"{metric}: precision={avg_score['precision']:.4f}, recall={avg_score['recall']:.4f}, fmeasure={avg_score['fmeasure']:.4f}")

# Calculate perplexity
all_input_ids = torch.cat([example["input_ids"] for example in test_dataset])
perplexity = perplexity_for_llama(model, device, tokenizer)
print("Perplexity:", perplexity.item())
