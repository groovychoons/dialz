from dataclasses import dataclass
from IPython.display import HTML

import torch
from transformers import AutoTokenizer
from typing import Union, List

def html_value_to_color(val: float, vmin: float, vmax: float) -> str:
    """
    Map a scalar score to a CSS RGB color string on a red→white→turquoise gradient.

    :param val: Activation score.
    :param vmin: Minimum score (mapped to red).
    :param vmax: Maximum score (mapped to turquoise).
    :return: CSS `rgb(r,g,b)` string.
    """
    norm = (val - vmin) / (vmax - vmin)
    if norm < 0.5:
        ratio = norm / 0.5
        r, g, b = 255, int(102 + 153 * ratio), int(102 + 153 * ratio)
    else:
        ratio = (norm - 0.5) / 0.5
        r = int(255 - 191 * ratio)
        g = int(255 - 31 * ratio)
        b = int(255 - 47 * ratio)
    return f"rgb({r},{g},{b})"


def highlight_token(score: float, vmin: float = -1.0, vmax: float = 1.0) -> str:
    """
    Generate ANSI escape codes for 24‑bit background highlighting (red→white→turquoise)
    with black text. Uses the same color interpolation logic as html_value_to_color.

    :param score: Activation score.
    :param vmin: Minimum score (mapped to red).
    :param vmax: Maximum score (mapped to turquoise).
    :return: ANSI escape sequence string.
    """
    norm = (max(min(score, vmax), vmin) - vmin) / (vmax - vmin)
    if norm < 0.5:
        ratio = norm / 0.5
        r, g, b = 255, int(102 + 153 * ratio), int(102 + 153 * ratio)
    else:
        ratio = (norm - 0.5) / 0.5
        r = int(255 - 191 * ratio)
        g = int(255 - 31 * ratio)
        b = int(255 - 47 * ratio)
    return f"\033[48;2;{r};{g};{b}m\033[38;2;0;0;0m"


def visualize_activation(
    input_text: str,
    model: "SteeringModel",
    control_vector: "SteeringVector",
    layer_index: Union[int, List[int]] = None,
    mode: str = "ansi",
    show_score: bool = False
) -> Union[str, HTML]:
    """
    Highlight token activations by projecting hidden states onto a steering vector.

    Supports negative indices (e.g., -1 for final layer) and averaging over multiple layers.
    Outputs ANSI-colored text for terminal or HTML snippet for Jupyter notebooks.

    :param input_text: Original input string.
    :param model: SteeringModel instance wrapping an LLM.
    :param control_vector: SteeringVector defining the intervention direction.
    :param layer_index: Single index or list of indices; negative values count from end.
    :param mode: "ansi" for console output, "html" for notebook display.
    :param show_score: If True, append numeric score to each token.
    :return: ANSI string or IPython.display.HTML.
    :raises ValueError: If no controlled layers set on the model.
    :raises RuntimeError: If hidden states are not captured.
    """
    model.reset()

    # Determine which layers to use.
    if layer_index is None:
        if not model.layer_ids:
            raise ValueError("No controlled layers set on this model!")
        layer_index = model.layer_ids[-1]

    if not isinstance(layer_index, list):
        layers_to_use = [layer_index]
    else:
        layers_to_use = layer_index

    # Prepare a container to store hidden states.
    hook_states = {}

    # Define and register hook function for each layer.
    def get_hook_fn(key):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hook_states[key] = out[0]
            else:
                hook_states[key] = out
        return hook_fn

    # Retrieve the list of layers from the model.
    def model_layer_list(m):
        if hasattr(m, "model"):
            return m.model.layers
        elif hasattr(m, "transformer"):
            return m.transformer.h
        else:
            raise ValueError("Cannot locate layers for this model type")

    layers = model_layer_list(model.model)

    # Register hooks on each requested layer.
    hooks = []
    for li in layers_to_use:
        real_layer_idx = li if li >= 0 else len(layers) + li
        hook_handle = layers[real_layer_idx].register_forward_hook(get_hook_fn(li))
        hooks.append(hook_handle)

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    offsets = encoded["offset_mapping"][0].tolist()

    with torch.no_grad():
        _ = model.model(input_ids)
    for h in hooks:
        h.remove()

    if layers_to_use[0] not in hook_states:
        raise RuntimeError("Did not capture hidden states in the forward pass!")
    seq_len = hook_states[layers_to_use[0]].shape[1]
    aggregated = [0.0] * seq_len

    # For each requested layer, compute dot products and sum them.
    for idx in layers_to_use:
        hidden = hook_states[idx][0]
        # Use the provided index; if not found in control_vector.directions,
        # try using the real (non-negative) index.
        key_for_direction = idx if idx in control_vector.directions else (len(layers) + idx)
        direction = torch.tensor(
            control_vector.directions[key_for_direction],
            device=model.device,
            dtype=model.model.dtype,
        )
        for i in range(seq_len):
            aggregated[i] += (torch.dot(hidden[i+1], direction).item() if i+1 < seq_len else 0.0)
    avg_scores = [s / len(layers_to_use) for s in aggregated]
    max_abs = max(abs(s) for s in avg_scores) or 1.0

    if mode == "html":
        html = "<div style='white-space: pre-wrap; font-family: monospace; line-height:1.3;'>"
        for (start, end), score in zip(offsets, avg_scores):
            token = input_text[start:end] or " "
            bg = html_value_to_color(score, -max_abs, max_abs)
            label = f"{token} ({score:.2f})" if show_score else token
            html += (
                f"<span style='display:inline-block; background-color:{bg}; color:black; "
                f"padding:2px;'>"
                f"{label}</span>"
            )
        html += "</div>"
        return HTML(html)


    ansi_output = ""
    for (start, end), score in zip(offsets, avg_scores):
        token = input_text[start:end] or " "
        ansi_output += f"{highlight_token(score, -max_abs, max_abs)}{token}{(' ('+f'{score:.2f}'+')') if show_score else ''}\033[0m"
    return ansi_output
