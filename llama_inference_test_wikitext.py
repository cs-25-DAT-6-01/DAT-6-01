import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rouge_score import rouge_scorer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utility import perplexity
from utility import filter_lines
from collections import defaultdict

# Define file name and such
gpt_model_name = "openai-community-gpt2"
llama_model_name = "meta-llama-Llama-3.2-1B"
amount_of_epochs = "10"
alpha = "10"
lambd = "0.2"
beta = "0.5"
gamma = "1.0"
temperature = "1.5"

# Path to the trained model/tokenizer
#model_path = f"model-{gpt_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"
#tokenizer_path = f"model-{gpt_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

model_path = f"model-{llama_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"
tokenizer_path = f"model-{llama_model_name}_epochs-{amount_of_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading wikitext dataset")
# Example: Load a dataset like "wikitext"
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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

print("Starting tokenization")
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'])
#test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=2, pin_memory=True)

print(model)
print("Memory used (MBs):", model.get_memory_footprint() / 1e6)
model.eval()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
total_inference_time = 0
total_score = defaultdict(list)
for i in range(len(test_dataset)):
    input_ids = test_dataset[i]["input_ids"].unsqueeze(0).to(device)
    attention_mask = test_dataset[i]["attention_mask"].unsqueeze(0).to(device)
    reference_text = test_dataset[i]["text"]
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Start time
    start_time = time.time()
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

    new_output = tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, "")
    score = scorer.score(reference_text, new_output)
    for metric, score_values in score.items():
        total_score[metric].append(score_values)

average_inference_time = total_inference_time / len(test_dataset)
print("Average inference time (seconds):", average_inference_time)

all_input_ids = torch.cat([example["input_ids"] for example in test_dataset])
perplexity = perplexity(model, device, tokenizer)
print("Perplexity:", perplexity.item())

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
