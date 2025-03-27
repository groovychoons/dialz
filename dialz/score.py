import typing
import warnings

import torch
from transformers import AutoTokenizer

if typing.TYPE_CHECKING:
    from .vector import SteeringVector, SteeringModel


def get_activation_score(
    input_text: str,
    model: "SteeringModel",
    control_vector: "SteeringVector",
    layer_index=None,  # can be int or list of ints
    scoring_method: str = "default",  # 'default', 'last_token', 'max_token', or 'median_token'
) -> float:
    """
    Returns the activation score for the input_text by projecting hidden state(s)
    onto the given control_vector direction(s) for the specified layer(s). If
    multiple layers are provided, the activation scores are averaged.

    Scoring methods:
        - 'default': Average the dot products over all tokens.
        - 'last_token': Use only the dot product of the final token.
        - 'max_token': Use the maximum dot product value among all tokens.
        - 'median_token': Use the median of the dot product values among all tokens.

    :param input_text: The input string to evaluate.
    :param control_vector: A ControlVector containing direction(s) keyed by layer index.
    :param layer_index: An int or a list of ints representing the layer(s) to use.
                        If None, defaults to the last controlled layer in model.layer_ids.
    :param scoring_method: A string specifying which scoring method to use.
    :return: A single float representing the averaged activation score.
    """
    # 1) Reset the model to ensure no control is applied.
    model.reset()

    # 2) Determine the layer(s) to use.
    if layer_index is None:
        if not model.layer_ids:
            raise ValueError("No controlled layers set on this model!")
        layer_index = model.layer_ids[-1]

    # If a single int is provided, wrap it in a list for unified processing.
    if not isinstance(layer_index, list):
        layers_to_use = [layer_index]
    else:
        layers_to_use = layer_index

    # 3) Prepare a container to store hidden states for each requested layer.
    hook_states = {}

    # 4) Define and register a hook function for each layer.
    def get_hook_fn(key):
        def hook_fn(module, inp, out):
            # If out is a tuple (hidden, present, ...), take the first element.
            if isinstance(out, tuple):
                hook_states[key] = out[0]
            else:
                hook_states[key] = out

        return hook_fn

    # 5) Retrieve the list of layers from the model.
    def model_layer_list(m):
        if hasattr(m, "model"):
            return m.model.layers
        elif hasattr(m, "transformer"):
            return m.transformer.h
        else:
            raise ValueError("Cannot locate layers for this model type")

    layers = model_layer_list(model.model)

    # 6) For each provided layer index, compute its actual index and register the hook.
    hooks = []
    for li in layers_to_use:
        real_layer_idx = li if li >= 0 else len(layers) + li
        hook_handle = layers[real_layer_idx].register_forward_hook(get_hook_fn(li))
        hooks.append(hook_handle)

    # 7) Build a tokenizer from the model name.
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # 8) Encode the input text and perform a forward pass.
    encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        _ = model.model(input_ids)

    # 9) Remove hooks to clean up.
    for hook in hooks:
        hook.remove()

    # 10) For each layer, compute the activation score using the chosen scoring method.
    scores = []
    for li in layers_to_use:
        if li not in hook_states:
            raise RuntimeError(
                f"Did not capture hidden states for layer {li} in the forward pass!"
            )
        # Extract hidden states for the single batch: shape [seq_len, hidden_dim]
        hidden_states = hook_states[li][0]
        # Retrieve the corresponding direction from the control_vector.
        if li not in control_vector.directions:
            raise ValueError(f"No direction for layer {li} in control_vector!")
        direction_np = control_vector.directions[li]
        direction = torch.tensor(
            direction_np, device=model.device, dtype=model.model.dtype
        )

        # Compute dot products for all tokens: shape [seq_len]
        dot_vals = hidden_states @ direction

        # Determine score based on the scoring_method.
        if scoring_method == "default":
            # Average over all tokens.
            score_tensor = dot_vals.mean()
        elif scoring_method == "last_token":
            # Use only the final token.
            score_tensor = dot_vals[-1]
        elif scoring_method == "max_token":
            # Use the maximum token's dot product.
            score_tensor = dot_vals.max()
        elif scoring_method == "median_token":
            # Use the median token's dot product.
            score_tensor = dot_vals.median()
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method}")

        scores.append(score_tensor.item())

    # 11) Average the scores across the selected layers.
    avg_score = sum(scores) / len(scores)
    return avg_score
