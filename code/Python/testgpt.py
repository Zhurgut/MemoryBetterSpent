import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def compute_zero_shot_ppl(split="test", stride=512):
    # 1) Load & filter empty lines
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    raw = raw.filter(lambda ex: ex["text"].strip() != "")

    # 2) Concatenate into one long string
    all_text = "\n\n".join(raw["text"])

    # 3) Tokenize entire stream (no truncation)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encodings = tokenizer(all_text, return_tensors="pt")
    input_ids = encodings.input_ids.to("cuda" if torch.cuda.is_available() else "cpu")[0]

    # 4) Prepare model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(input_ids.device)

    # 5) Sliding window
    W = model.config.n_positions     # 1024
    n_tokens = input_ids.size(0)
    nlls = []
    total_scored = 0

    for i in range(0, n_tokens, stride):
        start = max(i + stride - W, 0)
        end   = min(i + stride, n_tokens)
        chunk = input_ids[start:end].unsqueeze(0)   # [1, L]

        tgt_len = end - i
        labels = chunk.clone()
        # mask out the context portion
        labels[:, :-tgt_len] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=labels)
            # outputs.loss is averaged over the non-ignored positions
            # multiply by tgt_len to get total NLL for this window
            nlls.append(outputs.loss * tgt_len)
            total_scored += tgt_len

    # 6) Final perplexity
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_scored)
    return ppl.item()

if __name__ == "__main__":
    print("Zero-shot PPL:", compute_zero_shot_ppl())