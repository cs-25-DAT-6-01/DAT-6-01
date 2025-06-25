import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2LMHeadModel,
    BitsAndBytesConfig,
)
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

from loss_functions import prototype_log_loss

from torch.utils.data import Dataset, DataLoader

import time

# === CONFIG ===
HL_TOKEN = "[HL]"
MAX_QUESTION_LENGTH = 32
MAX_CONTEXT_LENGTH = 128
MAX_INPUT_LENGTH = MAX_CONTEXT_LENGTH + MAX_QUESTION_LENGTH
CHECKPOINT_DIR = "./checkpoints"
LOSS_LOG_PATH = "./loss_curve.json"
MODEL_SAVE_PATH = "./student_qg_model"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# === TOKENIZER ===
def get_tokenizer(base_model):
    # Load the tokenizer for the specified base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Add special tokens for highlighting answers
    tokenizer.add_tokens([HL_TOKEN], special_tokens=True)
    # Set the padding side to left for the tokenizer
    tokenizer.padding_side = "left"
    # Set the pad token to the end-of-sequence token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        # Add a separator token if not present
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if tokenizer.eos_token is None:
        # Add an end-of-sequence token if not present
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})
    return tokenizer


# === DATA ===
# Function to highlight the answer in the context
def highlight_answer(context: str, answer: str) -> str:
    start_idx = context.find(answer)
    if start_idx == -1:
        raise ValueError("Answer not found in context")
    end_idx = start_idx + len(answer)
    return context[:start_idx] + HL_TOKEN + answer + HL_TOKEN + context[end_idx:]

# Dataset class for the SQuAD dataset
class SquadDataset(Dataset):
    def __init__(self, tokenizer):
        # Load the SQuAD dataset and select a subset for training
        self.data = load_dataset("squad")["train"].select(range(10000))
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Get the item at the specified index
        item = self.data[idx]
        # Extract context, answer, and question from the item
        context = item["context"]
        answer = item["answers"]["text"][0]
        question = item["question"]
        # Highlight the answer in the context and tokenize the input
        hl_context = highlight_answer(context, answer) + self.tokenizer.sep_token
        # Tokenize the highlighted context
        tokens = self.tokenizer(
            hl_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
        )
        # Return the tokenized input along with the context, answer, and question
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "context": context,
            "answer": answer,
            "question": question,
        }

    def __len__(self):
        return len(self.data)


def train():
    # Login to Hugging Face Hub, using the HF_TOKEN environment variable
    login(os.getenv("HF_TOKEN"))
    
    # Set the quantization configuration for the models
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Define the model names for the student and teacher models
    student_model_name = "meta-llama/Llama-3.2-1B"
    teacher_model_name = "meta-llama/Llama-3.1-8B"

    # Load the teacher and student models with the specified quantization configuration
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, device_map="auto", torch_dtype="auto", quantization_config=bnb_config)
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name, device_map="auto", torch_dtype="auto", quantization_config=bnb_config)
    tokenizer = get_tokenizer(teacher_model_name)

    # Resize the token embeddings of the student and teacher models to match the tokenizer
    student_model.resize_token_embeddings(len(tokenizer))
    teacher_model.resize_token_embeddings(len(tokenizer))

    # Set the pad token ID for both models to the tokenizer's pad token ID
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.config.pad_token_id = tokenizer.pad_token_id

    # Set the end-of-sequence token ID for both models to the tokenizer's end-of-sequence token ID
    student_first_device = list(student_model.hf_device_map.values())[0]
    teacher_first_device = list(teacher_model.hf_device_map.values())[0]

    # Set the teacher model to evaluation mode
    teacher_model.eval()

    # Tokenize the SQuAD dataset
    dataset = SquadDataset(tokenizer)
    # Create a DataLoader for the dataset with specified batch size and other parameters
    train_dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    print("Starting tokenization")

    # Initialize the optimizer for the student model
    # Using AdamW optimizer with a learning rate of 5e-5
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    # Set the number of epochs for training
    num_epochs = 10

    # Plotting metrics
    loss_history = []

    print("Starting training")
    for epoch in range(num_epochs):
        # Distillation configuration parameters
        # These parameters can be adjusted based on the distillation strategy
        alpha = 8
        lambd = 0.7
        beta = 0.3
        gamma = 1.5
        temperature = 2
        student_model.train()

        # Initialize total loss for the epoch and start time
        total_loss = 0
        epoch_start = time.time()

        # Iterate over the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Move input IDs and attention mask to the student's first device
            input_ids = batch["input_ids"].to(student_first_device)
            attention_mask = batch["attention_mask"].to(student_first_device)

            # Compute the teacher's logits without gradients
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids.to(teacher_first_device),
                    attention_mask=attention_mask.to(teacher_first_device),
                ).logits

            # Compute the student's logits
            student_logits = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # Compute the loss using the prototype log loss function
            loss, kl, align, toptok, entropy = prototype_log_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_first_device=student_first_device,
                return_components=True,
            )

            # Every 100 steps, print the loss and its components
            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}: "
                    f"Loss={loss.item():.2f} | KL={kl.item():.2f} | Align={align.item():.2f} | "
                    f"TopTok={toptok.item():.4f} | Entropy={entropy.item():.2f}"
                )

            # Backpropagate the loss and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the total loss for the epoch
            total_loss += loss.item()

        # Compute the average loss for the epoch and the time taken for the epoch
        epoch_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
        )

        # Append the epoch loss to the loss history for plotting
        loss_history.append(epoch_loss)

    print("Saving model")
    # Save the student model and tokenizer
    model_name = student_model_name.replace("/", "-")
    out_dir = f"model-{model_name}_epochs-{num_epochs}_squad_alpha-{alpha}_beta-{beta}_lambd-{lambd}_gamma-{gamma}_temperature-{temperature}"

    student_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Plot the loss history
    plot_metrics(
        metrics={"loss": loss_history},
        run_tag="squad",
        out_dir=out_dir,
    )


def main():
    train()


if __name__ == "__main__":
    main()
