import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
import torch.distributed as dist
from torcheval.metrics import Perplexity as Perplexity
from torch.utils import checkpoint
from sentence_transformers import SentenceTransformer
from utility import plot_metrics
from utility import filter_lines

from loss_functions import prototype_log_loss


def train():
    login(os.getenv("HF_TOKEN"))

    print("Loading openai-community/gpt2-large")
    # Load the pre-trained openai-community/gpt2-large (teacher)
    teacher_model_name = "openai-community/gpt2-large"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"
    # teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, device_map="auto", torch_dtype="auto"
    )

    print("Loading gpt2 model")
    # Load the pre-trained "openai-community/gpt2" model (student)
    student_model_name = "openai-community/gpt2"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "left"
    # student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name, device_map="auto", torch_dtype="auto"
    )

    print("Loading wikitext dataset")
    # Example: Load a dataset like "wikitext"
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_dataset = train_dataset.map(
        lambda example: {"text": filter_lines(example["text"])}
    )
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)
    train_dataset = train_dataset.select(range(10000))
    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    print("Starting tokenization")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    # test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # DataLoader for the dataset
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, num_workers=4, pin_memory=True
    )
    # test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    student_first_device = list(student_model.hf_device_map.values())[0]
    teacher_first_device = list(teacher_model.hf_device_map.values())[0]

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    num_epochs = 10

    # Plotting metrics
    loss_history = []
    ppl_history = []

    print("Starting training")
    for epoch in range(num_epochs):
        alpha = 8
        lambd = 0.7
        beta = 0.3
        gamma = 1.5
        temperature = 2
        student_model.train()

        total_loss = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_dataloader):
            perplexity_metric = Perplexity().to(student_first_device)

            input_ids = batch["input_ids"].to(student_first_device)
            attention_mask = batch["attention_mask"].to(student_first_device)
            labels = input_ids.clone().detach()

            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids.to(teacher_first_device),
                    attention_mask=attention_mask.to(teacher_first_device),
                ).logits

            student_logits = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            loss, kl, align, toptok, entropy = prototype_log_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_first_device=student_first_device,
                return_components=True,
            )

            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}: "
                    f"Loss={loss.item():.2f} | KL={kl.item():.2f} | Align={align.item():.2f} | "
                    f"TopTok={toptok.item():.4f} | Entropy={entropy.item():.2f}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_probs = F.log_softmax(student_logits, dim=-1)
            perplexity_metric.update(log_probs, labels)
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        perplexity_score = perplexity_metric.compute()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Perplexity: {perplexity_score}, Time: {epoch_time:.2f}s"
        )

        loss_history.append(epoch_loss)
        ppl_history.append(perplexity_score)

    print("Saving model")
    # Save the student model and tokenizer
    model_name = student_model_name.replace("/", "-")
    out_dir = f"model-{model_name}_epochs-{num_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

    student_model.save_pretrained(out_dir)
    student_tokenizer.save_pretrained(out_dir)

    plot_metrics(
        metrics={"loss": loss_history, "perplexity": ppl_history},
        run_tag="wikitext",
        out_dir=out_dir,
    )


def main():
    train()


if __name__ == "__main__":
    main()
