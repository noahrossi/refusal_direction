import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Gemma 4 uses a new chat template ("<|turn>user\n...<turn|>\n<|turn>model\n...")
# and forces an empty <|channel>thought<channel|> block before the assistant's
# response (the jinja template ignores enable_thinking=False and always emits
# the thought-block prefix). The HF tokenizer for this model does NOT auto-add
# <bos>; it is instead included in the chat template string.
GEMMA4_CHAT_TEMPLATE = (
    "<bos><|turn>user\n{instruction}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
)

GEMMA4_CHAT_TEMPLATE_WITH_SYSTEM = (
    "<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{instruction}<turn|>\n"
    "<|turn>model\n<|channel>thought\n<channel|>"
)

# Token id of "I" (no leading space) under the Gemma 4 tokenizer — the typical
# first token of an English refusal ("I cannot...", "I'm sorry..."). Verified
# against google/gemma-4-31B-it via tok.encode('I', add_special_tokens=False).
GEMMA4_REFUSAL_TOKS = [236777]


def format_instruction_gemma4_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = GEMMA4_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, system=system
        )
    else:
        formatted_instruction = GEMMA4_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_gemma4_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma4_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma4_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    # add_special_tokens=False because the template already contains <bos>.
    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )


def _text_model(model):
    # Gemma4ForConditionalGeneration → .model (Gemma4Model) → .language_model (Gemma4TextModel)
    return model.model.language_model


def orthogonalize_gemma4_weights(model, direction: Float[Tensor, "d_model"]):
    text = _text_model(model)
    text.embed_tokens.weight.data = get_orthogonalized_matrix(
        text.embed_tokens.weight.data, direction
    )

    for block in text.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_gemma4_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    text = _text_model(model)
    dtype = text.layers[layer - 1].mlp.down_proj.weight.dtype
    device = text.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    text.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Gemma4Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        # The multimodal Gemma4Config keeps text-decoder params under .text_config.
        # The pipeline's generate_directions reads model.config.num_hidden_layers /
        # hidden_size directly, so mirror them onto the top-level config.
        text_cfg = model.config.text_config
        model.config.num_hidden_layers = text_cfg.num_hidden_layers
        model.config.hidden_size = text_cfg.hidden_size

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_gemma4_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            GEMMA4_CHAT_TEMPLATE.split("{instruction}")[-1],
            add_special_tokens=False,
        )

    def _get_refusal_toks(self):
        return GEMMA4_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return _text_model(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma4_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_gemma4_weights, direction=direction, coeff=coeff, layer=layer
        )
