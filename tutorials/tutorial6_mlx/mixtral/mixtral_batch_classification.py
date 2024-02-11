# Copyright © 2023 Apple Inc.
# Modifications Copyright © 2024 Reexpress AI, Inc.
# See the Reexpress AI tutorial for usage instructions.

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from sentencepiece import SentencePieceProcessor

import codecs
import time

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    moe: dict = None


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class RoPE(nn.RoPE):
    def __init__(self, dims: int, traditional: bool = False):
        super().__init__(dims, traditional)

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=1000000, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = RoPE(args.head_dim, traditional=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]
        self.experts = [FeedForward(args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)
        inds = mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt[None, :])
        y = mx.concatenate(y)

        return y.reshape(orig_shape)


class MOETransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = MOEFeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Mixtral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [MOETransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        standard_output=False
    ):

        h = self.tok_embeddings(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        if standard_output:
            return self.output(self.norm(h[:, T - 1: T, :])), cache
        else:
            all_layers_norm = self.norm(h)
            max_mask_factor = -1e9

            all_layers_norm_output = self.output(all_layers_norm)
            logit_norm_constant = 100
            max_prob_vocab_indexes = mx.argmax(all_layers_norm_output, axis=2, keepdims=True)
            max_prob = mx.take_along_axis(all_layers_norm_output, max_prob_vocab_indexes, axis=-1)
            mask_array_max_prob = mx.ones_like(all_layers_norm_output)
            mask_array_max_prob[:, mx.arange(T), max_prob_vocab_indexes.squeeze()] = max_mask_factor
            second_max_prob = \
                mx.max(mask_array_max_prob*all_layers_norm_output, axis=2, keepdims=True)[0, :, 0] / logit_norm_constant
            max_prob = max_prob[0, :, 0] / logit_norm_constant

            no_idx = 1770
            yes_idx = 5592
            return all_layers_norm_output[:, T - 1: T, :], cache, all_layers_norm[:, T - 1: T, :], \
                mx.mean(all_layers_norm, axis=1, keepdims=True), T, \
                max_prob, second_max_prob, max_prob - second_max_prob, \
                all_layers_norm_output[:, T - 1, yes_idx] / logit_norm_constant,\
                all_layers_norm_output[:, T - 1, no_idx] / logit_norm_constant


def get_data(filename_with_path):
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out


def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weight_files = glob.glob(str(model_path / "weights.*.npz"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())
    weights = tree_unflatten(list(weights.items()))
    model = Mixtral(model_args)
    if quantization is not None:
        # TODO: Quantize gate matrices when < 32 tiles supported
        quantization["linear_class_predicate"] = (
            lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8
        )
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.update(weights)
    return model, tokenizer


def standard_generate(prompt: mx.array, model: Mixtral, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    logits, cache = model(prompt[None], standard_output=True)
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(y[:, None], cache, standard_output=True)
        y = sample(logits.squeeze(1))
        yield y


def generate(prompt: mx.array, model: Mixtral):

    logits, cache, final_token_hidden_layer, hidden_layer_average, T, max_prob, second_max_prob, top_prob_diff, \
        yes_logit, no_logit = model(prompt[None])
    y = mx.argmax(logits[:, -1, :], axis=-1)
    yield y, final_token_hidden_layer, hidden_layer_average, T, max_prob, second_max_prob, top_prob_diff, \
        yes_logit, no_logit

    while True:
        logits, cache, final_token_hidden_layer, hidden_layer_average, T, max_prob, second_max_prob, top_prob_diff,\
            yes_logit, no_logit = model(y[:, None], cache)
        y = mx.argmax(logits.squeeze(1), axis=-1)
        yield y, final_token_hidden_layer, hidden_layer_average, T, max_prob, second_max_prob, top_prob_diff, \
            yes_logit, no_logit


def get_document_attributes(model, document_string="", prompt_text="", trailing_instruction="",
                            full_input_tokenized=None, return_input_length=False):
    if full_input_tokenized is None:
        if prompt_text == "":
            print(f"The prompt is blank. Exiting.")
            exit()
        if trailing_instruction == "":
            print(f"The trailing instruction is blank. Exiting.")
            exit()
        full_input = f"[INST] How can I help you?\n{prompt_text} {document_string} {trailing_instruction} Yes or No? [/INST]"
        full_input_tokenized = mx.array(tokenizer.encode(full_input))

    document_attributes = []
    gen_out = next(generate(full_input_tokenized, model))

    final_token_hidden_layer, hidden_layer_average, T, max_prob, second_max_prob, top_prob_diff, yes_logit, no_logit = \
        gen_out[1], gen_out[2], gen_out[3], gen_out[4], gen_out[5], gen_out[6], gen_out[7], gen_out[8]

    document_attributes.append(mx.max(max_prob, keepdims=True))
    document_attributes.append(mx.min(max_prob, keepdims=True))
    document_attributes.append(mx.mean(max_prob, keepdims=True))

    document_attributes.append(mx.max(second_max_prob, keepdims=True))
    document_attributes.append(mx.min(second_max_prob, keepdims=True))
    document_attributes.append(mx.mean(second_max_prob, keepdims=True))

    document_attributes.append(mx.max(top_prob_diff, keepdims=True))
    document_attributes.append(mx.min(top_prob_diff, keepdims=True))
    document_attributes.append(mx.mean(top_prob_diff, keepdims=True))

    hidden_avg_split = mx.split(hidden_layer_average, indices_or_sections=8, axis=2)
    for hidden_group in hidden_avg_split:
        document_attributes.append( mx.mean(hidden_group, 2)[0] )
    final_hidden_split = mx.split(final_token_hidden_layer, indices_or_sections=8, axis=2)
    for hidden_group in final_hidden_split:
        document_attributes.append( mx.mean(hidden_group, 2)[0] )

    # yes/no generation
    document_attributes.append(max_prob[-1:]-yes_logit)
    document_attributes.append(max_prob[-1:]-no_logit)

    document_attributes.append(max_prob[-1:])
    document_attributes.append(second_max_prob[-1:])
    document_attributes.append(top_prob_diff[-1:])
    document_attributes.append(yes_logit)
    document_attributes.append(no_logit)

    # Handle Infinity/-Infinity with a simple clamp. NaN also becomes -2 for the purposes here. This, as with the
    # other renorm constants, should be modified for other models and use cases. (Here, we are assuming a max length
    # of around 5000 characters. For longer documents, the overflow/underflow may need to be handled differently.)
    document_attributes = mx.clip(mx.concatenate(document_attributes, 0).astype(mx.float32), -2.0, 2.0).tolist()
    if return_input_length:
        return document_attributes, full_input_tokenized.shape[0]
    return document_attributes


def construct_minimal_json_object(id_string, label, document, attributes, prompt="", info="", group=""):
    """
    Optional fields that are empty strings are dropped. In this case, attributes are expected to be present for every
    document; however, an empty list is not included in the JSON.
    """
    # required properties
    json_obj = {"id": id_string, "label": label,
                "document": document}
    # optional properties
    if len(attributes) > 0:
        json_obj["attributes"] = attributes
    if len(prompt) > 0:
        json_obj["prompt"] = prompt
    if len(info) > 0:
        json_obj["info"] = info
    if len(group) > 0:
        json_obj["group"] = group

    return json_obj


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixtral inference script and simple demo of generating binary re-ask "
                                                 "attributes for input into Reexpress (macOS application).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model_quantized",
        help="The path to the model weights, tokenizer, and config",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    parser.add_argument(
        "--input_filename", required=True,
        help="Input JSON lines file, in the standard Reexpress document format, with the exception that the "
             "prompt is not restricted to 250 characters. However, if the prompt is longer than 250 characters, "
             "consider using a shorter prompt when input into Reexpress; use the option "
             "--combine_the_prompt_and_document_fields_in_json_output; or if applicable, "
             "--drop_the_prompt_in_json_output.")
    parser.add_argument(
        "--output_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl. The new JSON lines will be appended (rather than "
             "overwritten) to this file.")
    parser.add_argument(
        "--prompt_text", default="",
        help="Added before the document. E.g., "
             "'Please classify the sentiment of the following review. Review:'")
    parser.add_argument(
        "--trailing_instruction", default="",
        help="Added after the prompt_text and document and before 'Yes or No?'. E.g., "
             "'Question: Does the previous document have a positive sentiment?'")

    parser.add_argument('--use_json_prompt', action='store_true',
                        help="If provided, --prompt_text is ignored, and the prompt in the JSON lines file for "
                             "each document is used.")
    parser.add_argument('--combine_the_prompt_and_document_fields_in_json_output', action='store_true',
                        help="If provided, the prompt and the document text are combined in the JSON lines file in "
                             "the 'document' field. Use this if the prompt is greater than 250 characters and you "
                             "want the Reexpress models to see the full prompt text.")
    parser.add_argument('--drop_the_prompt_in_json_output', action='store_true',
                        help="If provided, no 'prompt' field is included in the JSON output for each document. Use "
                             "with caution, but this can be a viable option if the prompt used for the Gen AI model "
                             "is a long instruction that is constant across all examples.")

    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("[INFO] Loading model from disk.")
    model, tokenizer = load_model(args.model_path)
    assert model.args.dim == 4096
    kMaxPythonChars = 4800
    input_data_json_list = get_data(args.input_filename)

    if args.combine_the_prompt_and_document_fields_in_json_output and args.drop_the_prompt_in_json_output:
        print(f"Only one of --combine_the_prompt_and_document_fields_in_json_output and "
              f"--drop_the_prompt_in_json_output can be provided. Exiting.")
        exit()

    start_time = time.time()
    total_tokens = 0
    doc_index = 0
    for one_document_json in input_data_json_list:
        prompt_text = args.prompt_text
        if args.use_json_prompt:
            if "prompt" in one_document_json:
                prompt_text = one_document_json["prompt"]
            else:
                print(f"The option --use_json_prompt was provided, but the document JSON lacks a 'prompt' field. "
                      f"Exiting.")
                exit()
        document_attributes, num_tokens = get_document_attributes(model, one_document_json['document'].strip(),
                                                                  prompt_text,
                                                                  args.trailing_instruction, return_input_length=True)
        total_tokens += num_tokens
        group_field_text = ""
        info_field_text = ""
        if "group" in one_document_json:
            group_field_text = one_document_json["group"]
        if "info" in one_document_json:
            info_field_text = one_document_json["info"]
        if args.combine_the_prompt_and_document_fields_in_json_output:
            document_text = f"{prompt_text} {one_document_json['document'].strip()}"
            prompt_text = ""
        elif args.drop_the_prompt_in_json_output:
            document_text = f"{one_document_json['document'].strip()}"
            prompt_text = ""
        else:
            document_text = f"{one_document_json['document'].strip()}"

        json_obj = construct_minimal_json_object(one_document_json["id"], one_document_json["label"],
                                                 document_text[0:kMaxPythonChars].strip(),
                                                 document_attributes, prompt=prompt_text,
                                                 info=info_field_text, group=group_field_text)

        save_by_appending_json_lines(args.output_jsonl_file, [json_obj])
        print(f"Saved document {doc_index} of {len(input_data_json_list)}")
        doc_index += 1

        if doc_index % 100 == 0:
            elapsed_seconds = time.time() - start_time
            print(f"Total tokens: {total_tokens}; total documents: {doc_index}")
            print(f"Elapsed seconds: {elapsed_seconds}")
            print(f"Tokens per second: {float(total_tokens) / elapsed_seconds}")
            print(f"Documents per second: {float(doc_index) / elapsed_seconds}")
            print("-"*60)
            print("")

    elapsed_seconds = time.time() - start_time
    print(f"++COMPLETE++")
    print(f"Total tokens: {total_tokens}; total documents: {doc_index}")
    print(f"Elapsed seconds: {elapsed_seconds}")
    print(f"Tokens per second: {float(total_tokens) / elapsed_seconds}")
    print(f"Documents per second: {float(doc_index) / elapsed_seconds}")