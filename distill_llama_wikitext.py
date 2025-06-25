import os
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from torcheval.metrics import Perplexity as Perplexity
from sentence_transformers import SentenceTransformer
from utility import plot_metrics
from utility import filter_lines
from loss_functions import prototype_log_loss, new_distillation_loss


def train():
    # Get the Hugging Face token from environment variable
    login(os.getenv("HF_TOKEN"))

    # Define the BitsAndBytesConfig for quantization
    # This config allows loading models in 8-bit precision with CPU offloading
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Load the teacher and student models and tokenizers
    # Using Llama-3.1-8B as the teacher and Llama-3.2-1B as the student
    teacher_model_name = "meta-llama/Llama-3.1-8B"
    student_model_name = "meta-llama/Llama-3.2-1B"
    embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Load the teacher model and tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # Set the pad token to the end-of-sequence token for the teacher tokenizer
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # Set the padding side to left for the teacher tokenizer
    teacher_tokenizer.padding_side = "left"
    # Load the teacher model with quantization and set it to evaluation mode
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    # Set the pad token ID to the end-of-sequence token ID for the teacher model
    teacher_model.config.pad_token_id = teacher_model.config.eos_token_id
    # Set the teacher model to evaluation mode
    teacher_model.eval()

    # Load the student model and tokenizer
    # Set the pad token to the end-of-sequence token for the student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    # Set the pad token to the end-of-sequence token for the student tokenizer
    student_tokenizer.pad_token = student_tokenizer.eos_token
    # Set the padding side to left for the student tokenizer
    student_tokenizer.padding_side = "left"
    # Load the student model with quantization and set it to evaluation mode
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    # Set the pad token ID to the end-of-sequence token ID for the student model
    student_model.config.pad_token_id = student_model.config.eos_token_id
    
    # Load the embedder model for computing similarity
    embedder = SentenceTransformer(embedder_model_name)
    # Set the embedder model to evaluation mode
    embedder.eval()

    # Load the Wikitext-2 dataset and preprocess it
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_dataset = train_dataset.map(
        lambda example: {"text": filter_lines(example["text"])}
    )
    # Filter out empty lines and select a subset of the dataset for training
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)
    train_dataset = train_dataset.select(range(10000))

    # Define a function to tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # Tokenize the dataset using the tokenize_function
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    # Set the format of the dataset to PyTorch tensors and specify the columns to keep
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create a DataLoader for the training dataset
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, num_workers=2, pin_memory=True
    )

    # Get the first device of the student and teacher models
    student_first_device = list(student_model.hf_device_map.values())[0]
    teacher_first_device = list(teacher_model.hf_device_map.values())[0]

    # Load optimizer for the student model
    # Using AdamW optimizer with a learning rate of 5e-5
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    # Set the number of epochs for training
    num_epochs = 10

    # Initialize lists to store loss and perplexity history
    loss_history = []
    ppl_history = []

    # Start the training loop
    for epoch in range(num_epochs):
        # Distillation configuration parameters
        alpha = 8
        lambd = 0.7
        beta = 0.3
        gamma = 1.5
        temperature = 2
        student_model.train()

        # Initialize the total loss for the epoch
        # and the start time for the epoch
        total_loss = 0
        epoch_start = time.time()

        # Iterate over the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Move the batch to the device of the student model
            perplexity_metric = Perplexity().to(student_first_device)

            # Move input_ids and attention_mask to the device of the student model
            input_ids = batch["input_ids"].to(student_first_device)
            attention_mask = batch["attention_mask"].to(student_first_device)
            # Clone input_ids to create labels for the loss computation, when using wikitext, we can use input_ids as labels
            labels = input_ids.clone().detach()

            # Compute the teacher model's logits
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids.to(teacher_first_device),
                    attention_mask=attention_mask.to(teacher_first_device),
                ).logits

            # Compute the student model's logits
            student_logits = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            # Compute the distillation loss using the prototype_log_loss function
            # This function computes the loss components: KL divergence, alignment, top token loss, and entropy
            loss, kl, align, toptok, entropy = prototype_log_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_first_device=student_first_device,
                return_components=True,
            )

            # Print the loss components every 100 steps
            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}: "
                    f"Loss={loss.item():.2f} | KL={kl.item():.2f} | Align={align.item():.2f} | "
                    f"TopTok={toptok.item():.4f} | Entropy={entropy.item():.2f}"
                )

            # Optimizer zero_grad, backward pass, and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the log probabilities for perplexity metric
            log_probs = F.log_softmax(student_logits, dim=-1)
            perplexity_metric.update(log_probs, labels)
            total_loss += loss.item()

        # Compute the average loss for the epoch and the perplexity score
        epoch_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        perplexity_score = perplexity_metric.compute().item()

        # Print the epoch loss, perplexity score, and time taken for the epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Perplexity: {perplexity_score}, Time: {epoch_time:.2f}s"
        )

        # Append the epoch loss and perplexity score to the history lists
        loss_history.append(epoch_loss)
        ppl_history.append(perplexity_score)

    # Model name and output directory for saving the trained student model
    model_name = student_model_name.replace("/", "-")
    out_dir = f"model-{model_name}_epochs-{num_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

    # Save model + tokenizer to the output directory
    student_model.save_pretrained(out_dir)
    student_tokenizer.save_pretrained(out_dir)

    # Call plot_metrics to visualize the training metrics
    plot_metrics(
        metrics={"loss": loss_history, "perplexity": ppl_history},
        run_tag="wikitext",
        out_dir=out_dir,
    )


def main():
    train()


if __name__ == "__main__":
    main()
