#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kør SQuAD-evaluation med timing på alle 1 000 eksempler.
Forudsætter:
  pip install transformers datasets evaluate torch --upgrade
og at din fine-tunede model + tokenizer ligger lokalt i
  model-<navn>_epochs-<n>_squad_alpha-<α>_beta-<β>_lambd-<λ>_gamma-<γ>_temperature-<T>
"""

import os
import time
import torch
from datasets import load_dataset
from transformers import (
    GPT2ForQuestionAnswering,
    AutoTokenizer,
    pipeline,
    Pipeline,
)
from evaluate import evaluator


class TimingPipeline(Pipeline):
    def __init__(self, wrapped_pipeline):
        self.wrapped = wrapped_pipeline
        self.inference_times = []

        super().__init__(
            model=wrapped_pipeline.model,
            tokenizer=wrapped_pipeline.tokenizer,
            task=wrapped_pipeline.task,
            device=wrapped_pipeline.device,
        )

    def _call(self, *args, **kwargs):
        start = time.time()
        output = self.wrapped(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.inference_times.append(time.time() - start)
        return output

    def __getattr__(self, item):
        return getattr(self.wrapped, item)


model_name = "openai-community-gpt2"
amount_of_epochs = "10"
alpha, lambd = "8", "0.7"
beta, gamma = "0.3", "1.5"
temperature = "2"

model_path = (
    f"model-{model_name}_epochs-{amount_of_epochs}_squad_"
    f"alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"
)
tokenizer_path = model_path

model = GPT2ForQuestionAnswering.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    local_files_only=True,
)

qa_pipeline = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer,
    doc_stride=128,
)

timed_qa_pipeline = TimingPipeline(qa_pipeline)

test_dataset = load_dataset("squad", split="validation[:1000]")
qa_evaluator = evaluator("question-answering")

eval_results = qa_evaluator.compute(
    model_or_pipeline=timed_qa_pipeline,
    data=test_dataset,
    metric="squad",
    batch_size=16,
    strategy="bootstrap",
    n_resamples=15,
)


times = timed_qa_pipeline.inference_times
print(f"Antal eksempler evalueret : {len(times)}")
print(f"Gennemsnitlig latency     : {sum(times) / len(times):.4f} sek")
print(f"Første 10 latencies       : {times[:10]}")
print("\nEvaluation metrics:")
for k, v in eval_results.items():
    print(f"  {k:10s}: {v}")
