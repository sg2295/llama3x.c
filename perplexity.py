"""
Perplexity = exp(mean cross-entropy loss). Lower is better.
Conventions match llama.cpp perplexity tool.
Usage:
  python3 perplexity.py <model_path> -d path/to/eval.txt
"""

import argparse
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from export import load_hf_model
from tokenizer import get_tokenizer


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize_text(tokenizer, text: str, add_bos: bool = True) -> List[int]:
    return tokenizer.encode(
        text,
        bos=add_bos,
        eos=False,
        allowed_special="all",
        disallowed_special=(),
    )


def chunk_tokens(
    ids: List[int],
    n_ctx: int,
    ppl_stride: int,
) -> List[List[int]]:
    """
    Split token list into chunks of length n_ctx.
    - If ppl_stride == 0: non-overlapping chunks (stride = n_ctx).
    - If ppl_stride > 0: strided overlapping chunks (stride = ppl_stride).
    """
    chunks: List[List[int]] = []
    stride = n_ctx if ppl_stride == 0 else ppl_stride
    if ppl_stride == 0:
        # Non-overlapping: [0:n_ctx], [n_ctx:2*n_ctx], ...
        for start in range(0, len(ids) - n_ctx + 1, n_ctx):
            chunk = ids[start : start + n_ctx]
            if len(chunk) == n_ctx:
                chunks.append(chunk)
    else:
        # Strided: start = 0, stride, 2*stride, ...
        for start in range(0, max(1, len(ids) - n_ctx + 1), stride):
            end = start + n_ctx
            if end > len(ids):
                break
            chunk = ids[start:end]
            if len(chunk) == n_ctx:
                chunks.append(chunk)
    return chunks


@torch.inference_mode()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset_path: str,
    n_ctx: int = 512,
    ppl_stride: int = 0,
    batch_size: int = 2048,
    device: Optional[torch.device] = None,
    n_chunks: int = -1,
) -> Tuple[float, float]:
    """
    Compute perplexity. Matches llama.cpp behavior:
    - ppl_stride == 0: non-overlapping chunks, score only second half of each chunk.
    - ppl_stride > 0: strided chunks, score only last ppl_stride positions per chunk.
    """
    text = load_text(dataset_path)
    ids = tokenize_text(tokenizer, text)
    min_tokens = 2 * n_ctx if ppl_stride == 0 else n_ctx + 1
    if len(ids) < min_tokens:
        raise ValueError(
            f"Not enough tokens in {dataset_path}: got {len(ids)}, need at least {min_tokens} "
            f"for n_ctx={n_ctx}, ppl_stride={ppl_stride}. Use a longer file or smaller --ctx-size."
        )
    chunks = chunk_tokens(ids, n_ctx, ppl_stride)
    if not chunks:
        raise ValueError(
            f"No chunks produced from {dataset_path} for n_ctx={n_ctx}, ppl_stride={ppl_stride}."
        )
    if n_chunks >= 0:
        chunks = chunks[:n_chunks]
    n_chunk = len(chunks)
    print(f"Calculating perplexity over {n_chunk} chunks, batch_size={batch_size} (tokens)")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Which positions to score (llama.cpp convention):
    # ppl_stride == 0: second half only (first = n_ctx/2)
    # ppl_stride > 0: last ppl_stride positions only
    if ppl_stride == 0:
        first_scored = n_ctx // 2  # score positions first..n_ctx-2
    else:
        first_scored = n_ctx - 1 - ppl_stride  # score last ppl_stride positions

    n_chunks_per_fwd = max(1, batch_size // n_ctx)
    for i in range(0, len(chunks), n_chunks_per_fwd):
        batch_chunks = chunks[i : i + n_chunks_per_fwd]
        act_n_chunks_fwd = len(batch_chunks)
        tokens = torch.tensor(batch_chunks, dtype=torch.long, device=device)
        # targets[t] = next token after tokens[t]; last position ignored
        targets = torch.cat(
            [tokens[:, 1:], torch.full((act_n_chunks_fwd, 1), -1, device=device, dtype=torch.long)], dim=1
        )
        targets[:, :first_scored] = -1
        logits = model(tokens, targets=targets)
        assert logits is not None, "Logits are None"
        for c in range(act_n_chunks_fwd):
            loss_c = F.cross_entropy(
                logits[c].view(-1, logits.size(-1)),
                targets[c].view(-1),
                ignore_index=-1,
            )
            n_c = (targets[c] != -1).sum().item()
            total_loss += loss_c.item() * n_c
            total_tokens += n_c
            # Instead of showing per-chunk ppl, show cumulative result so far
            # ppl_chunk = math.exp(loss_c.item()) if loss_c.item() < 1e2 else float("inf")
            # print(f"[{i + c + 1}]{ppl_chunk:.4f}", end=" ", flush=True)
            ppl_cumulative = (
                math.exp(total_loss / total_tokens)
                if total_tokens and total_loss / total_tokens < 1e2
                else float("inf")
            )
            print(f"[{i + c + 1}]{ppl_cumulative:.4f}", end=" ", flush=True)
    print("")
    mean_loss = total_loss / total_tokens if total_tokens else float("nan")
    ppl = math.exp(mean_loss) if mean_loss < 1e2 else float("inf")
    return ppl, mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity")
    parser.add_argument("model_path", type=str, help="HuggingFace model path")
    parser.add_argument("-d", "--data",type=str,required=True,help="Path to eval text")
    parser.add_argument("-c", "--ctx-size",type=int,default=512,help="Chunk length")
    parser.add_argument("--ppl-stride",type=int,default=0,help="Stride for perplexity; 0 = non-overlapping chunks, >0 = strided")
    parser.add_argument("-b", "--batch-size", type=int, default=2048, help="Total number of tokens to run in one pass")
    parser.add_argument("--chunks", type=int, default=-1, help="Max number of chunks to process (-1 = all)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"], help="Model dtype")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        parser.error(f"data file not found: {args.data}")

    if args.ppl_stride < 0:
        parser.error("--ppl-stride must be >= 0")

    if args.ppl_stride > 0 and args.ppl_stride >= args.ctx_size:
        parser.error("--ppl-stride must be < --ctx-size when > 0")

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = load_hf_model(args.model_path)
    if model is None:
        parser.error("Failed to load model.")
    model.to(device=device, dtype=dtype)

    tokenizer = get_tokenizer(args.model_path)

    ppl, loss = compute_perplexity(
        model, tokenizer, args.data, args.ctx_size, args.ppl_stride, args.batch_size, device, args.chunks
    )
    print(f"Loss:  {loss:.4f}")
    print(f"PPL:   {ppl:.4f}")
