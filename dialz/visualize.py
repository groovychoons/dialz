from IPython.display import HTML, display
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass


def html_value_to_color(val: float, vmin: float, vmax: float) -> str:
    """
    Map a score to a red→white→turquoise CSS color.
    
    For a value of vmin the color is a vibrant red (#FF6666), at 0 it is white,
    and at vmax it transitions to a turquoise (#40E0D0).
    """
    # Normalize value into [0, 1]
    norm = (val - vmin) / (vmax - vmin)
    if norm < 0.5:
        # Interpolate from red (#FF6666) to white (#FFFFFF)
        ratio = norm / 0.5
        r = 255
        g = int(102 + 153 * ratio)
        b = int(102 + 153 * ratio)
    else:
        # Interpolate from white (#FFFFFF) to turquoise (#40E0D0)
        ratio = (norm - 0.5) / 0.5
        r = int(255 - 191 * ratio)  # 255 -> 64
        g = int(255 - 31 * ratio)   # 255 -> 224
        b = int(255 - 47 * ratio)   # 255 -> 208
    return f"rgb({r},{g},{b})"

def highlight_token(score: float) -> str:
    """
    Generate an ANSI escape code to highlight the background of a token
    with black text. Uses a simple interpolation for the background.
    """
    s = max(-1.0, min(1.0, score))
    if s >= 0.0:
        r = int(255 * (1.0 - s))
        g = int(255 * (1.0 - s))
        b = 255
    else:
        fraction = s + 1.0
        r = 255
        g = int(255 * fraction)
        b = int(255 * fraction)
    return f"\033[48;2;{r};{g};{b}m\033[38;2;0;0;0m"

def visualize_activation(
    input_text: str,
    model: "SteeringModel",
    control_vector: "SteeringVector",
    layer_index: int = None,
    mode: str = "ansi",  # "ansi" or "html"
    show_score: bool = False
) -> str | HTML:
    """
    Visualize the model's activations by highlighting the background behind each token,
    based on the dot product of the hidden state of token (i+1) and a control vector.

    The output can be rendered as ANSI escape codes (for console use) or as HTML (for
    Jupyter notebooks). In HTML mode, a red→white→turquoise gradient is used with
    black text.

    :param input_text: The original input text.
    :param model: The steering model used for evaluation.
    :param control_vector: The steering vector (control direction) to project onto.
    :param layer_index: The index of the layer from which to extract hidden states.
                        If not provided, the last layer in model.layer_ids is used.
    :param mode: "ansi" for terminal output, "html" for an HTML snippet.
    :param show_score: If True, display the numerical score alongside each token.
    :return: A string (ANSI) or an IPython.display.HTML object with the highlighted text.
    """
    # Reset the model so no control is applied.
    model.reset()

    # Choose the layer to hook. Use the last controlled layer if not provided.
    if layer_index is None:
        if not model.layer_ids:
            raise ValueError("No layer_ids set on this model!")
        layer_index = model.layer_ids[-1]

    @dataclass
    class HookState:
        hidden: torch.Tensor = None  # Shape: [batch, seq_len, hidden_dim]

    hook_state = HookState()

    def hook_fn(module, inp, out):
        """
        A forward hook to capture hidden states from the selected layer.
        """
        hook_state.hidden = out[0] if isinstance(out, tuple) else out

    def model_layer_list(m):
        """
        Retrieve the list of layers from the model.
        """
        if hasattr(m, "model"):
            return m.model.layers
        elif hasattr(m, "transformer"):
            return m.transformer.h
        else:
            raise ValueError("Cannot locate layers for this model type")

    layers = model_layer_list(model.model)
    real_idx = layer_index if layer_index >= 0 else len(layers) + layer_index
    hook_handle = layers[real_idx].register_forward_hook(hook_fn)

    # Tokenize the input text and obtain offset mappings.
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    offsets = encoded["offset_mapping"][0].tolist()  # List of (start, end) pairs

    # Forward pass to capture hidden states.
    with torch.no_grad():
        _ = model.model(input_ids)
    hook_handle.remove()

    if hook_state.hidden is None:
        raise RuntimeError("Did not capture hidden states in the forward pass!")
    hidden = hook_state.hidden[0]

    # Retrieve the control direction.
    if layer_index not in control_vector.directions:
        raise ValueError(f"No direction for layer {layer_index} in the control vector!")
    direction = torch.tensor(
        control_vector.directions[layer_index],
        device=model.device,
        dtype=hidden.dtype,
    )

    # Compute dot products for each token using the hidden state of token i+1.
    seq_len = hidden.size(0)
    scores = []
    for i in range(seq_len):
        next_idx = i + 1
        dot_val = torch.dot(hidden[next_idx], direction).item() if next_idx < seq_len else 0.0
        scores.append(dot_val)
    max_abs = max(abs(s) for s in scores) or 1.0

    if mode == "html":
        html = "<div style='white-space: pre-wrap; font-family: monospace;'>"
        for (start, end), score in zip(offsets, scores):
            token = input_text[start:end] or " "
            bg = html_value_to_color(score, -max_abs, max_abs)
            label = f"{token} ({score:.2f})" if show_score else token
            html += f"<span style='background-color: {bg}; color: black; padding: 2px 3px;'>{label}</span>"
        html += "</div>"
        return HTML(html)
    else:
        # ANSI mode
        ansi_output = ""
        for (start, end), score in zip(offsets, scores):
            token = input_text[start:end] or " "
            ansi_bg = highlight_token(score)
            reset = "\033[0m"
            label = f"{token} ({score:.2f})" if show_score else token
            ansi_output += f"{ansi_bg}{label}{reset}"
        return ansi_output
