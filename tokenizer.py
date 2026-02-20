# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import argparse
import os
import struct
from pathlib import Path
from typing import List

import tiktoken
from tiktoken.load import load_tiktoken_bpe

TOKENIZER_MODEL = "tokenizer.model"  # the llama tiktoken tokenizer model


class Tokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str=TOKENIZER_MODEL):
        assert os.path.isfile(model_path), model_path
        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        num_base_tokens = len(mergeable_ranks)
        num_reserved_special_tokens = 256

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_words = self.model.n_vocab
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self, s: str, bos: bool, eos: bool, allowed_special, disallowed_special
    ) -> List[int]:
        assert type(s) is str
        t = self.model.encode(
            s,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t)

    def export(self, tokenizer_bin: str):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.model.decode_single_token_bytes(i)
            s = i
            tokens.append(t)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        with open(tokenizer_bin, "wb") as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

def get_tokenizer(hf_model_path: str) -> Tokenizer:
    # Grab tokenizer from hf repo
    candidates = [
        os.path.join(hf_model_path, "original", "tokenizer.model"),
        os.path.join(hf_model_path, "tokenizer.model"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return Tokenizer(path)

    raise ValueError(f"No tokenizer found in {hf_model_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_hf_model", type=str, help="input huggingface model path")
    parser.add_argument("output_tokenizer_bin", type=str, help="output tokenizer binary path", default="tokenizer.bin")

    args = parser.parse_args()

    t = get_tokenizer(args.input_hf_model)
    t.export(args.output_tokenizer_bin)
