import os
import numbers
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
import re
from torch.cuda.amp import autocast


def plot_metrics(
    metrics: dict[str, list | tuple],
    run_tag: str = "run",
    out_dir: str = "plots",
    show: bool = False,
):
    if not metrics:
        raise ValueError("`metrics` dict is empty.")

    lengths = {len(v) for v in metrics.values()}
    if len(lengths) != 1:
        raise ValueError("All metric value sequences must have the same length.")
    for name, values in metrics.items():
        if not all(isinstance(x, numbers.Number) for x in values):
            raise TypeError(f"Metric '{name}' contains non‑numeric values.")

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, next(iter(lengths)) + 1)

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
        
def filter_lines(text):
  """Filters out lines starting and ending with ===, ==, or =."""
  lines = text.split('\n')
  filtered_lines = [line for line in lines if not re.fullmatch(r"^\s*[=]{1,3}\s*.*\s*[=]{1,3}\s*$", line)]  
  return '\n\n'.join(filtered_lines)

def perplexity(model, device, tokenizer):
    torch.cuda.empty_cache()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    print(f"Total sequence length: {seq_len}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            with autocast():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl

def perplexity_for_llama(model, device, tokenizer):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    first_device = list(model.hf_device_map.values())[0]

    #max_length = model.config.max_position_embeddings
    max_length = 1024
    stride = 256
    seq_len = encodings.input_ids.size(1)
    print(f"Total sequence length: {seq_len}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(first_device)
        target_ids = input_ids.clone().to(first_device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl