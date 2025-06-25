import os
import numbers
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
import re
from torch.cuda.amp import autocast

# Plot training metrics such as loss or accuracy over epochs
def plot_metrics(
    metrics: dict[str, list | tuple],
    run_tag: str = "run",
    out_dir: str = "plots",
    show: bool = False,
):
    # Check if metrics dictionary is empty
    if not metrics:
        raise ValueError("`metrics` dict is empty.")

    # Ensure all metric value lists have the same length (e.g., number of epochs)
    lengths = {len(v) for v in metrics.values()}
    if len(lengths) != 1:
        raise ValueError("All metric value sequences must have the same length.")

    # Validate that all metric values are numeric
    for name, values in metrics.items():
        if not all(isinstance(x, numbers.Number) for x in values):
            raise TypeError(f"Metric '{name}' contains non‑numeric values.")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, next(iter(lengths)) + 1)

    # Plot and save each metric
    for metric_name, values in metrics.items():
        plt.figure()
        plt.plot(epochs, values, marker="o")
        plt.title(f"{metric_name.capitalize()} – {run_tag}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)

        fname = os.path.join(out_dir, f"{run_tag}_{metric_name}.png")
        plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


# Remove lines enclosed by 1–3 equal signs on both ends (e.g., "== Header ==")
def filter_lines(text):
    """Filters out lines starting and ending with ===, ==, or =."""
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not re.fullmatch(r"^\s*[=]{1,3}\s*.*\s*[=]{1,3}\s*$", line)]
    return '\n\n'.join(filtered_lines)


# Compute perplexity of a model using Wikitext-2 (for general transformer models)
def perplexity(model, device, tokenizer):
    torch.cuda.empty_cache()

    # Load test split of Wikitext-2
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions  # model's max input length
    stride = 512  # overlap between windows
    seq_len = encodings.input_ids.size(1)
    print(f"Total sequence length: {seq_len}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    # Slide a window over the dataset
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # length of target segment

        # Prepare input and target
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask tokens not to be used for loss

        with torch.no_grad():
            with autocast():  # Automatic mixed precision
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss  # CrossEntropyLoss

        # Count valid tokens and accumulate loss
        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # adjust for label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood
    ppl = torch.exp(avg_nll)  # perplexity = exp(nll)
    return ppl


# Compute perplexity for LLaMA models (requires device mapping)
def perplexity_for_llama(model, device, tokenizer):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    first_device = list(model.hf_device_map.values())[0]  # use first mapped device

    max_length = 1024  # Fixed for LLaMA models
    stride = 256
    seq_len = encodings.input_ids.size(1)
    print(f"Total sequence length: {seq_len}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(first_device)
        target_ids = input_ids.clone().to(first_device)
        target_ids[:, :-trg_len] = -100  # Mask tokens outside target window

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl
