import torch
import torch.nn.functional as F


def prototype_log_loss(
    student_logits,
    teacher_logits,
    student_first_device,
    epsilon=1e-6,
    alpha=1.0,   # Weight for the cosine alignment term
    lambd=1.0,   # Weight for the KL divergence term
    beta=1.0,    # Weight for the max token penalty
    gamma=1.7,   # Weight for the entropy penalty
    temperature=2,  # Softmax temperature for distillation
    return_components=False,  # Optionally return individual loss terms
):
    # Apply temperature scaling to logits
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits.detach() / temperature  # Detach to avoid backprop through teacher

    # Compute log softmax and softmax probabilities
    log_student_probs = F.log_softmax(student_logits, dim=-1).to(student_first_device)
    teacher_probs = F.softmax(teacher_logits, dim=-1).to(student_first_device)

    # KL divergence between teacher and student (scaled by temperature^2)
    kl_term = (temperature**2) * F.kl_div(
        log_student_probs, teacher_probs, reduction="batchmean"
    )

    # Recover student probabilities from log space
    student_probs = log_student_probs.exp()

    # Compute cosine similarity between teacher and student probability distributions
    dot = torch.sum(student_probs * teacher_probs, dim=-1)  # dot product
    student_norm = torch.norm(student_probs, dim=-1)
    teacher_norm = torch.norm(teacher_probs, dim=-1)
    cos_sim = dot / (student_norm * teacher_norm + epsilon)  # avoid divide-by-zero

    # Logarithmic cosine alignment loss (encourages directional similarity)
    attraction_term = torch.mean(torch.log(1 + cos_sim + epsilon))

    # Max-token penalty: squared difference of top token confidence
    top_token_penalty = (
        (
            torch.max(teacher_probs, dim=-1).values
            - torch.max(student_probs, dim=-1).values
        )
        .pow(2)
        .mean()
    )

    # Entropy of the student’s output (controls certainty)
    entropy = -torch.sum(student_probs * log_student_probs, dim=-1)
    entropy_penalty = entropy.mean()

    # Total composite loss
    loss = (
        lambd * kl_term                  # Divergence matching
        - alpha * attraction_term       # Directional alignment (note: we subtract it)
        + beta * top_token_penalty      # Penalize over/under-confidence on top token
        + gamma * entropy_penalty       # Regularize uncertainty
    )

    # Optionally return individual components for logging/debugging
    if return_components:
        return loss, kl_term, attraction_term, top_token_penalty, entropy_penalty
    return loss



# def new_distillation_loss(
#     alpha,
#     beta,
#     student,
#     teacher,
#     tokenizer,
#     embedder,
#     gen_config,
#     batch,
#     student_first_device,
#     teacher_first_device,
# ):
#     # Teacher-forced CE
#     with torch.no_grad():
#         teacher_outputs = teacher.generate(
#             input_ids=batch["input_ids"].to(teacher_first_device),
#             attention_mask=batch["attention_mask"].to(teacher_first_device),
#             generation_config=gen_config,
#         )

#     teacher_texts = [
#         tokenizer.decode(out, skip_special_tokens=True) for out in teacher_outputs
#     ]
#     teacher_inputs = tokenizer(
#         teacher_texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=128,
#     ).to(student_first_device)
#     student_logits = student(
#         teacher_inputs.input_ids, attention_mask=teacher_inputs.attention_mask
#     ).logits

#     shift_logits = student_logits[..., :-1, :].contiguous()
#     shift_labels = teacher_inputs.input_ids[..., 1:].contiguous()
#     min_len = min(shift_logits.size(1), shift_labels.size(1))
#     shift_logits = shift_logits[:, :min_len, :]
#     shift_labels = shift_labels[:, :min_len]
#     loss_ce = F.cross_entropy(
#         shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
#     )

#     # Embedding MSE
#     with torch.no_grad():
#         teacher_embeddings = embedder.encode(teacher_texts, convert_to_tensor=True).to(
#             student_first_device
#         )

#     student_generated = student.generate(
#         input_ids=batch["input_ids"].to(student_first_device),
#         attention_mask=batch["attention_mask"].to(student_first_device),
#         generation_config=gen_config,
#     )

#     student_texts = [
#         tokenizer.decode(out, skip_special_tokens=True) for out in student_generated
#     ]
#     student_embeddings = embedder.encode(student_texts, convert_to_tensor=True).to(
#         student_first_device
#     )
#     loss_embed = F.mse_loss(
#         student_embeddings.to(student_first_device),
#         teacher_embeddings.to(student_first_device),
#     )

#     # Consistency CE
#     student_free_inputs = tokenizer(
#         student_texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=128,
#     ).to(student_first_device)
#     free_logits = student(
#         student_free_inputs.input_ids, attention_mask=student_free_inputs.attention_mask
#     ).logits
#     shift_free_logits = free_logits[..., :-1, :].contiguous()
#     shift_teacher_labels = teacher_inputs.input_ids[..., 1:].contiguous()
#     min_len2 = min(shift_free_logits.size(1), shift_teacher_labels.size(1))
#     shift_free_logits = shift_free_logits[:, :min_len2, :]
#     shift_teacher_labels = shift_teacher_labels[:, :min_len2]

#     loss_consistency = F.cross_entropy(
#         shift_free_logits.reshape(-1, shift_free_logits.size(-1)),
#         shift_teacher_labels.reshape(-1).to(student_first_device),
#     )

#     total_loss = (
#         loss_ce.to(student_first_device)
#         + alpha * loss_embed.to(student_first_device)
#         + beta * loss_consistency.to(student_first_device)
#     )
#     return total_loss


# def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
#     """
#     Computes the knowledge distillation loss.

#     :param student_logits: Output logits from the student model
#     :param teacher_logits: Output logits from the teacher model
#     :param true_labels: Ground truth labels
#     :param T: Temperature (scaling factor for softening logits)
#     :param alpha: Weight for combining cross-entropy and KL divergence losses
#     """
#     # Cross entropy loss for the student model on the true labels
#     ce_loss = F.cross_entropy(
#         student_logits.view(-1, student_logits.size(-1)), true_labels.view(-1)
#     )

#     # Soft targets (teacher's soft output)
#     soft_teacher_output = F.softmax(teacher_logits / T, dim=-1)
#     soft_student_output = F.log_softmax(student_logits / T, dim=-1)

#     # KL Divergence loss
#     kl_loss = F.kl_div(soft_student_output, soft_teacher_output, reduction="batchmean")

#     # Combine losses with weighting
#     return alpha * ce_loss + (1 - alpha) * (T * T) * kl_loss
