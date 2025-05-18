import time
import torch
from transformers import pipeline, AutoTokenizer, GPT2ForQuestionAnswering, LlamaForCausalLM
from datasets import load_dataset
from evaluate import evaluator
import string

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

def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))

def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(true_tokens)
    return 2 * prec * rec / (prec + rec)

model_name = "meta-llama-Llama-3.2-1B"
amount_of_epochs = "10"
alpha = "8"
lambd = "0.7"
beta = "0.3"
gamma = "1.5"
temperature = "2"

model_path = f"model-{model_name}_epochs-{amount_of_epochs}_squad_alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"
tokenizer_path = model_path

model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

#qa_pipeline = pipeline(
#    task="text-generation",
#    model=model,
#    tokenizer=tokenizer,
#)

test_dataset = load_dataset("squad", split="validation[:1000]")

#qa_evaluator = evaluator("question-answering")
#eval_results = qa_evaluator.compute(
#    model_or_pipeline=qa_pipeline,
#    data=test_dataset,
#    metric="squad"
#)

inference_times = []
results = []
for example in test_dataset:
    prompt = f"Context: {example['context']}\nAnswer: {example['answers']}\nQuestion:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    inference_times.append(end - start)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append(generated_text)

exact_matches = []
f1_scores = []
for example, pred in zip(test_dataset, results):
    true_answer = example["questions"]["text"][0] if example["questions"]["text"] else ""
    exact_matches.append(compute_exact(pred, true_answer))
    f1_scores.append(compute_f1(pred, true_answer))

print("Inference times (seconds):", inference_times)
average_inference_time = sum(inference_times) / len(inference_times)
print("Average inference time (seconds):", average_inference_time)
print("Average Exact Match:", sum(exact_matches) / len(exact_matches))
print("Average F1 Score:", sum(f1_scores) / len(f1_scores))
