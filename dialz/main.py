import os
from dotenv import load_dotenv

# from .vector import get_vector
from .dataset import Dataset
from .model import ControlModel
from .vector import ControlVector

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

dataset = Dataset.create_dataset(
    model_name, ["feminist", "sexist"], "starters", num_sents=200
)
print("here")
model = ControlModel(model_name, list(range(-5, -18, -1)), hf_token)
racism_vector = ControlVector.train(model, dataset)

from transformers import AutoTokenizer

list_of_strings = []

tokenizer = AutoTokenizer.from_pretrained(model_name, token=model.token)
tokenizer.pad_token_id = 0


def generate_with_vector(
    input: str,
    vector: ControlVector,
    coeffs: tuple[float, float],
    max_new_tokens: int = 20,
    repetition_penalty: float = 1.1,
    show_baseline: bool = True,
):
    positive_coeff, negative_coeff = coeffs
    assert positive_coeff > 0
    assert negative_coeff < 0

    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    if show_baseline:
        print("==baseline ---------------------------------------------------")
        model.reset()
        print(
            tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
        )
        list_of_strings.append(
            tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
        )

    print("\n++control ---------------------------------------------------")
    model.set_control(vector, positive_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
    list_of_strings.append(
        tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
    )

    print("\n--control ---------------------------------------------------")
    model.set_control(vector, negative_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
    list_of_strings.append(
        tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
    )

    model.reset()

    with open(f"output_{input[:10]}.txt", "w") as f:
        for string in list_of_strings:
            f.write(string + "\n")


highlighted_output = model.visualize_activation(
    input_text="Good CEOs are all men. Men tend to be better at leadership roles.",
    control_vector=racism_vector,
)

print(highlighted_output)

highlighted_output = model.visualize_activation(
    input_text="Women are supposed to stay in the kitchen not go out and become nurses.",
    control_vector=racism_vector,
)
print(highlighted_output)

highlighted_output = model.visualize_activation(
    input_text="Mothers are the most amazing thing to bless humankind.",
    control_vector=racism_vector,
)
print(highlighted_output)

print(
    model.get_activation_score("Female empowerment is an amazing thing.", racism_vector)
)
