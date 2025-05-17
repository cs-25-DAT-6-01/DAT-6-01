import time
import torch
from transformers import pipeline, AutoTokenizer, GPT2ForQuestionAnswering, LlamaForQuestionAnswering
from datasets import load_dataset
from evaluate import evaluator

model_name = "meta-llama-Llama-3.2-1B"
amount_of_epochs = "10"
alpha = "8"
lambd = "0.7"
beta = "0.3"
gamma = "1.5"
temperature = "2"

model_path = f"model-{model_name}_epochs-{amount_of_epochs}_squad_alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"
tokenizer_path = model_path

model = LlamaForQuestionAnswering.from_pretrained(model_path, local_files_only=True, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

qa_pipeline = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer,
)

test_dataset = load_dataset("squad", split="validation[:1000]")

qa_evaluator = evaluator("question-answering")
eval_results = qa_evaluator.compute(
    model_or_pipeline=qa_pipeline,
    data=test_dataset,
    metric="squad"
)

inference_times = []
for example in test_dataset:
    start = time.time()
    _ = qa_pipeline(
        question=example["question"],
        context=example["context"],
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    inference_times.append(end - start)

print("Inference times (seconds):", inference_times)
average_inference_time = sum(inference_times) / len(inference_times)
print("Average inference time (seconds):", average_inference_time)
print("Evaluation results:", eval_results)
