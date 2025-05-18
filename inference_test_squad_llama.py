import time
import string
from statistics import mean

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evaluate import load


def _normalize(text: str) -> str:
    def remove_articles(t):
        return " ".join(w for w in t.split() if w.lower() not in {"a", "an", "the"})

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        table = str.maketrans("", "", string.punctuation)
        return t.translate(table)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def compute_exact(pred: str, truth: str) -> int:
    return int(_normalize(pred) == _normalize(truth))


def compute_f1(pred: str, truth: str) -> float:
    pred_tokens = _normalize(pred).split()
    truth_tokens = _normalize(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


MODEL_NAME = "meta-llama-Llama-3.2-1B"
EPOCHS = 10
ALPHA, LAMBDA, BETA, GAMMA, TEMP = 8, 0.7, 0.3, 1.5, 2
MODEL_PATH = (
    f"model-{MODEL_NAME}_epochs-{EPOCHS}_squad_"
    f"alpha-{ALPHA}_beta-{BETA}_lambd-{LAMBDA}_gamma-{GAMMA}_temperature-{TEMP}"
)

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

test_dataset = load_dataset("squad", split="validation[:1000]")

inference_times = []
pred_questions = []

for ex in test_dataset:
    prompt = f"Context: {ex['context']}\nAnswer: {ex['answers']['text']}\nQuestion:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    output_ids = model.generate(**inputs, max_new_tokens=64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_times.append(time.time() - start)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    question = generated.split("Question:")[-1].strip()
    pred_questions.append(question)

exact_matches = []
f1_scores = []

for ex, pred in zip(test_dataset, pred_questions):
    gold = ex["question"] or ""
    exact_matches.append(compute_exact(pred, gold))
    f1_scores.append(compute_f1(pred, gold))

bleu_metric = load("bleu")
bertscore_metric = load("bertscore")

references = [ex["question"] or "" for ex in test_dataset]

bleu = bleu_metric.compute(
    predictions=pred_questions, references=[[ref] for ref in references]
)
bertscore = bertscore_metric.compute(
    predictions=pred_questions, references=references, lang="en", batch_size=32
)

print("Average inference latency :", mean(inference_times))
print("Exact-Match               :", mean(exact_matches))
print("Token-level F1            :", mean(f1_scores))
print("Corpus BLEU-4             :", bleu["bleu"])
print("Mean BERTScore-F1         :", mean(bertscore["f1"]))
