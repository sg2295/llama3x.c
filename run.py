import argparse
import time

import torch

from export import load_hf_model
from tokenizer import get_tokenizer

def sample_next_token(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp == 0.0:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model_path", type=str, help="huggingface model path")
    parser.add_argument("--dtype", type=str, help="dtype of the model", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("-p", "--prompt", type=str, help="prompt to generate", default="In a galaxy")
    parser.add_argument("-n", "--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--device", type=str, default=None, help="Torch device")

    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = load_hf_model(args.hf_model_path)

    if model is None:
        parser.error("Can't load input model!")

    # Move model to device / dtype
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device} with dtype: {dtype}")
    model.to(device=device, dtype=dtype)

    tokenizer = get_tokenizer(args.hf_model_path)

    prompt = args.prompt

    # Run text generation
    input_ids = tokenizer.encode(
        prompt,
        bos=True,
        eos=False,
        allowed_special="all",
        disallowed_special=(),
    )
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Print prompt immediately, flush so it appears before generation
    print(prompt, end="", flush=True)

    ttft = None
    num_generated = 0

    start_time = time.perf_counter()

    with torch.inference_mode():
        # We don't use generate so that we can get TTFT and flush output
        for _ in range(args.max_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= model.params.max_seq_len
                else idx[:, -model.params.max_seq_len :]
            )
            logits = model(idx_cond)[:, -1, :]  # fwd pass

            idx_next = sample_next_token(logits, args.temperature)

            next_id = idx_next.item()

            if next_id in tokenizer.stop_tokens:
                break

            piece = tokenizer.decode([next_id])
            print(piece, end="", flush=True)
            num_generated += 1

            if ttft is None:
                ttft = time.perf_counter() - start_time

            idx = torch.cat((idx, idx_next), dim=1)

    if ttft is not None:
        elapsed = time.perf_counter() - start_time
        tok_s = num_generated / elapsed if elapsed > 0 else 0
        print(f"Processed tokens: {num_generated}")
        print(f"Time to first token: {ttft * 1000:.0f} ms")
        print(f"Achieved tok/s: {tok_s:.1f}")
