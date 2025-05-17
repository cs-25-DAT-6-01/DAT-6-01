import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2ForQuestionAnswering
)
import torch
import torch.nn.functional as F
from datasets import load_dataset
import torch.distributed as dist
from evaluate import evaluator

from torch.utils.data import Dataset, DataLoader
from transformers import pipeline

import time
    
class TimingPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.inference_times = []
        
    def __call__(self, *args, **kwargs):
        is_batch = isinstance(args[0], list) if args else False
        if is_batch:
            batch = args[0]
            start = time.time()
            results = self.pipeline(batch)
            torch.cuda.synchronize()
            end = time.time()
            batch_time = (end - start)
            # Distribute batch time equally to each example
            per_example_time = batch_time / len(batch)
            self.inference_times.extend([per_example_time] * len(batch))
            return results
        else:
            start = time.time()
            result = self.pipeline(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.time()
            self.inference_times.append(end - start)
            return result
    
    @property
    def task(self):
        return self.pipeline.task


# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "10"
alpha = "8"
lambd = "0.7"
beta = "0.3"
gamma = "1.5"
temperature = "2"

# Path to the trained model/tokenizer
model_path = f"model-{model_name}_epochs-{amount_of_epochs}_squad_alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}_squad_alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"

model = GPT2ForQuestionAnswering.from_pretrained(model_path, device_map="auto", torch_dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map="auto", local_files_only=True)

qa_pipeline = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer,
)
timed_qa_pipeline = TimingPipeline(qa_pipeline)
qa_evaluator = evaluator("question-answering")

first_device = list(model.hf_device_map.values())[0]

test_dataset = load_dataset("squad", split="validation[:1000]")

eval_results = qa_evaluator.compute(
    model_or_pipeline=timed_qa_pipeline,
    data=test_dataset,
    metric="squad",
    strategy="bootstrap",
    n_resamples=15,
)

inference_times = timed_qa_pipeline.inference_times
print("Inference times (seconds):", inference_times)
average_inference_time = sum(inference_times) / len(inference_times)
print("Average inference time (seconds):", average_inference_time)
print("Evaluation results:", eval_results)