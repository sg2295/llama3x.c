import argparse
import torch

import export
from tokenizer import get_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model_path", type=str, help="huggingface model path")
    parser.add_argument("--dtype", type=str, help="dtype of the model", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("-p", "--prompt", type=str, help="prompt to generate", default="In a galaxy")
    parser.add_argument("-n", "--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cuda, mps, cpu)")

    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = export.load_hf_model(args.hf_model_path)

    if model is None:
        parser.error("Can't load input model!")

    # Move model to device / dtype
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        y = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    out_ids = y[0].tolist()
    # Drop the prompt tokens and stop at first stop token
    gen_ids = []
    for tok in out_ids[len(input_ids) :]:
        if tok in tokenizer.stop_tokens:
            break
        gen_ids.append(tok)
    text = tokenizer.decode(gen_ids)
    output = prompt + text
    print(output)
