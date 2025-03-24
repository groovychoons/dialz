import dataclasses
import typing
import warnings

import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

if typing.TYPE_CHECKING:
    from .vector import SteeringVector


class SteeringModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(
        self, model_name: str, layer_ids: typing.Iterable[int], token: str = None
    ):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new SteeringModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, torch_dtype=torch.float16
        )
        self.token = token

        self.model = self.model.to(
            "cuda:0"
            if torch.cuda.is_available()
            else "mps:0" if torch.backends.mps.is_available() else "cpu"
        )

        layers = model_layer_list(self.model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, SteeringModule):
                layers[layer_id] = SteeringModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "SteeringVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `SteeringVector` for the layers this SteeringModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: SteeringModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def visualize_activation(
        self,
        input_text: str,
        control_vector: "SteeringVector",
        layer_index: int = None,
    ) -> str:
        """
        Uses offset mappings to preserve the exact original text spacing.
        Token i is colored by the alignment (dot product) from token (i+1).
        If i+1 is out of range, we default that token's color to white.
        """
        self.reset()

        # 2) Pick the layer to hook
        if layer_index is None:
            if not self.layer_ids:
                raise ValueError("No layer_ids set on this model!")
            layer_index = self.layer_ids[-1]

        # We'll store the hidden states from the chosen layer in a HookState.
        @dataclasses.dataclass
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

        layers = model_layer_list(self.model)
        real_layer_index = (
            layer_index if layer_index >= 0 else len(layers) + layer_index
        )
        layers[real_layer_index].register_forward_hook(hook_fn)

        # 3) Tokenize with offset mappings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            return_offsets_mapping=True,  # so we can preserve original text slices
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offset_mapping = encoded["offset_mapping"][
            0
        ].tolist()  # list of (start, end) for each token

        # 4) Forward pass to capture hidden states
        with torch.no_grad():
            _ = self.model(input_ids)

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
            direction_np, dtype=self.model.dtype, device=self.device
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

    def get_activation_score(
        self,
        input_text: str,
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
                            If None, defaults to the last controlled layer in self.layer_ids.
        :param scoring_method: A string specifying which scoring method to use.
        :return: A single float representing the averaged activation score.
        """
        # 1) Reset the model to ensure no control is applied.
        self.reset()

        # 2) Determine the layer(s) to use.
        if layer_index is None:
            if not self.layer_ids:
                raise ValueError("No controlled layers set on this model!")
            layer_index = self.layer_ids[-1]

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

        layers = model_layer_list(self.model)

        # 6) For each provided layer index, compute its actual index and register the hook.
        hooks = []
        for li in layers_to_use:
            real_layer_idx = li if li >= 0 else len(layers) + li
            hook_handle = layers[real_layer_idx].register_forward_hook(get_hook_fn(li))
            hooks.append(hook_handle)

        # 7) Build a tokenizer from the model name.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        # 8) Encode the input text and perform a forward pass.
        encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        with torch.no_grad():
            _ = self.model(input_ids)

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
                direction_np, device=self.device, dtype=self.model.dtype
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


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class SteeringModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(model: SteeringModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, SteeringModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")


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
