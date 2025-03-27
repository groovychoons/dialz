import typing
import warnings

import torch
from transformers import AutoTokenizer
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from .vector import SteeringVector, SteeringModel


def visualize_activation(
    input_text: str,
    model: "SteeringModel",
    control_vector: "SteeringVector",
    layer_index: int = None,
) -> str:
    """
    Uses offset mappings to preserve the exact original text spacing.
    Token i is colored by the alignment (dot product) from token (i+1).
    If i+1 is out of range, we default that token's color to white.
    """
    model.reset()

    # 2) Pick the layer to hook
    if layer_index is None:
        if not model.layer_ids:
            raise ValueError("No layer_ids set on this model!")
        layer_index = model.layer_ids[-1]

    # We'll store the hidden states from the chosen layer in a HookState.
    @dataclass
    class HookState:
        hidden: torch.Tensor = None  # shape [batch, seq_len, hidden_dim]

    hook_state = HookState()

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            hook_state.hidden = out[0]
        else:
            hook_state.hidden = out

    # Identify the layer module and attach a forward hook
    def model_layer_list(m):
        if hasattr(m, "model"):
            return m.model.layers
        elif hasattr(m, "transformer"):
            return m.transformer.h
        else:
            raise ValueError("Cannot locate layers for this model type")

    layers = model_layer_list(model.model)
    real_layer_index = (
        layer_index if layer_index >= 0 else len(layers) + layer_index
    )
    layers[real_layer_index].register_forward_hook(hook_fn)

    # 3) Tokenize with offset mappings
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        return_offsets_mapping=True,  # so we can preserve original text slices
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    offset_mapping = encoded["offset_mapping"][
        0
    ].tolist()  # list of (start, end) for each token

    # 4) Forward pass to capture hidden states
    with torch.no_grad():
        _ = model.model(input_ids)

    if hook_state.hidden is None:
        raise RuntimeError("Did not capture hidden states in the forward pass!")

    # shape: (seq_len, hidden_dim)
    hidden_states = hook_state.hidden[0]

    # 5) Get the raw direction from the control vector
    if layer_index not in control_vector.directions:
        raise ValueError(
            f"No direction for layer {layer_index} in the control vector!"
        )
    direction_np = control_vector.directions[layer_index]
    direction = torch.tensor(
        direction_np, dtype=model.model.dtype, device=model.device
    )

    # 6) Compute dot product for each token *with index+1*
    #    We'll store them in a list the same length as hidden_states.
    seq_len = hidden_states.size(0)
    scores = []
    for i in range(seq_len):
        next_idx = i + 1
        if next_idx < seq_len:
            dot_val = torch.dot(hidden_states[next_idx], direction).item()
        else:
            # If next_idx is out of range, default to 0 => white color
            dot_val = 0.0
        scores.append(dot_val)

    # 7) Build the final colored string using offset mappings
    colored_text = ""
    for (start, end), score in zip(offset_mapping, scores):
        substring = input_text[start:end]  # exact slice of the original text
        color_prefix = color_token(score)  # your existing gradient function
        reset_code = "\033[0m"
        colored_text += f"{color_prefix}{substring}{reset_code}"

    return colored_text




def color_token(score: float) -> str:
    """
    Returns the token colored with a 24-bit ANSI code that smoothly
    interpolates from:
        - score = -1 => (255, 0, 0)     [red]
        - score =  0 => (255, 255, 255) [white]
        - score = +1 => (0, 0, 255)     [blue]
    """

    # Clamp score to [-1, 1]
    s = max(-1.0, min(1.0, score))

    if s >= 0.0:
        # s in [0..1] goes from white -> blue
        r = int(255 * (1.0 - s))
        g = int(255 * (1.0 - s))
        b = 255
    else:
        # s in [-1..0) goes from red -> white
        fraction = s + 1.0
        r = 255
        g = int(255 * fraction)
        b = int(255 * fraction)

    return f"\033[38;2;{r};{g};{b}m"
