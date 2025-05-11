import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from torcheval.metrics import Perplexity as Perplexity
from sentence_transformers import SentenceTransformer
from utility import plot_metrics
from utility import filter_lines


def new_distillation_loss(alpha, beta,  student, teacher, tokenizer, embedder, gen_config, batch, student_first_device, teacher_first_device):    
        # Teacher-forced CE
        with torch.no_grad():
            teacher_outputs = teacher.generate(
                input_ids=batch["input_ids"].to(teacher_first_device), 
                attention_mask=batch["attention_mask"].to(teacher_first_device), 
                generation_config=gen_config
                )

        teacher_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in teacher_outputs]
        teacher_inputs = tokenizer(teacher_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(student_first_device)
        student_logits = student(teacher_inputs.input_ids, attention_mask=teacher_inputs.attention_mask).logits
        
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_inputs.input_ids[..., 1:].contiguous()
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        loss_ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Embedding MSE
        with torch.no_grad():
            teacher_embeddings = embedder.encode(teacher_texts, convert_to_tensor=True).to(student_first_device)

        student_generated = student.generate(
            input_ids=batch["input_ids"].to(student_first_device),
            attention_mask=batch["attention_mask"].to(student_first_device), 
            generation_config=gen_config)
        
        student_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in student_generated]
        student_embeddings = embedder.encode(student_texts, convert_to_tensor=True).to(student_first_device)
        loss_embed = F.mse_loss(student_embeddings.to(student_first_device), teacher_embeddings.to(student_first_device))
        
        # Consistency CE
        student_free_inputs = tokenizer(student_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(student_first_device)
        free_logits = student(student_free_inputs.input_ids, attention_mask=student_free_inputs.attention_mask).logits
        shift_free_logits = free_logits[..., :-1, :].contiguous()
        shift_teacher_labels = teacher_inputs.input_ids[..., 1:].contiguous()
        min_len2 = min(shift_free_logits.size(1), shift_teacher_labels.size(1))
        shift_free_logits = shift_free_logits[:, :min_len2, :]
        shift_teacher_labels = shift_teacher_labels[:, :min_len2]

        loss_consistency = F.cross_entropy(shift_free_logits.reshape(-1, shift_free_logits.size(-1)), shift_teacher_labels.reshape(-1).to(student_first_device))

        total_loss = loss_ce.to(student_first_device) + alpha * loss_embed.to(student_first_device) + beta * loss_consistency.to(student_first_device)
        return total_loss

def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    """
    Computes the knowledge distillation loss.

    :param student_logits: Output logits from the student model
    :param teacher_logits: Output logits from the teacher model
    :param true_labels: Ground truth labels
    :param T: Temperature (scaling factor for softening logits)
    :param alpha: Weight for combining cross-entropy and KL divergence losses
    """
    # Cross entropy loss for the student model on the true labels
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), true_labels.view(-1))

    # Soft targets (teacher's soft output)
    soft_teacher_output = F.softmax(teacher_logits / T, dim=-1)
    soft_student_output = F.log_softmax(student_logits / T, dim=-1)

    # KL Divergence loss
    kl_loss = F.kl_div(soft_student_output, soft_teacher_output, reduction='batchmean')

    # Combine losses with weighting
    return alpha * ce_loss + (1 - alpha) * (T * T) * kl_loss


def train():
    login(os.getenv("HF_TOKEN"))
    
    bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    )

    print("Loading Llama-3.1-8B model")
    # Load the pre-trained llama 8b (teacher)
    teacher_model_name = "meta-llama/Llama-3.1-8B"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, quantization_config=bnb_config, device_map="auto", torch_dtype="auto")
    teacher_model.config.pad_token_id = teacher_model.config.eos_token_id    

    print("Loading Llama-3.2-1B model")
    # Load the pre-trained "openai-community/gpt2" model (student)
    student_model_name = "meta-llama/Llama-3.2-1B"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name, quantization_config=bnb_config, device_map="auto", torch_dtype="auto")
    student_model.config.pad_token_id = student_model.config.eos_token_id
    
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    gen_config = GenerationConfig(
        repetition_penalty = 1.2,
        bos_token_id = student_tokenizer.bos_token_id,
        pad_token_id = student_tokenizer.pad_token_id,
    )

    print("Loading wikitext dataset")
    # Example: Load a dataset like "wikitext"
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_dataset = train_dataset.map(lambda example: {'text': filter_lines(example['text'])})
    train_dataset = train_dataset.filter(lambda example: len(example['text']) > 0)
    train_dataset = train_dataset.select(range(10000))
    #test_dataset = dataset["test"]

     # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=512)

    print("Starting tokenization")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    #test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    #test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)
    
    student_first_device = list(student_model.hf_device_map.values())[0]
    teacher_first_device = list(teacher_model.hf_device_map.values())[0]

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    num_epochs = 10
    
    #Plotting metrics
    loss_history = []
    ppl_history  = []

    print("Starting training")
    for epoch in range(num_epochs):
        alpha = 0.5
        beta = 0.5
        student_model.train()
        teacher_model.eval()  # Teacher model doesn't need gradient updates

        total_loss = 0
        for batch in train_dataloader:
            perplexity_metric = Perplexity().to(student_first_device)
            input_ids = batch["input_ids"].to(student_first_device)
            attention_mask = batch["attention_mask"].to(student_first_device)
            labels = input_ids.clone().detach()

            # Calculate distillation loss
            loss = new_distillation_loss(alpha, beta, student_model, teacher_model, teacher_tokenizer, embedder, gen_config, batch, student_first_device, teacher_first_device)
                        
            # Backward pass
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            
            # Forward pass through the student model for perplexity calculation
            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits, dim=-1).to(student_first_device)
            perplexity_metric.update(log_probs, labels)

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        perplexity_score = perplexity_metric.compute()
        print(f"Perplexity: {perplexity_score}")  
        
        loss_history.append(epoch_loss)
        ppl_history.append(perplexity_score)

    print("Saving model")
    # Save the student model and tokenizer
    model_name = student_model_name.replace("/", "-")
    student_model.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_webquestions_alpha-{alpha}_beta-{beta}")
    student_tokenizer.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_webquestions_alpha-{alpha}_beta-{beta}")
    
    plot_metrics(
        metrics={"loss": loss_history, "perplexity": ppl_history},
        run_tag="wikitext",
        out_dir=f"model-{model_name}_epochs-{num_epochs}_wikitext_alpha-{alpha}_beta-{beta}"
    )


def main():
    train()


if __name__ == "__main__":
    main()