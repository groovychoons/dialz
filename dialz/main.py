import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# from .vector import get_vector
from .dataset import Dataset
from .model import ControlModel
from .vector import ControlVector

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

corpus_dataset = Dataset.load_corpus("data")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token_id = 0

model = AutoModelForCausalLM.from_pretrained(
    model_name, token=hf_token, torch_dtype=torch.float16
)
model = model.to(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available() else "cpu"
)
model = ControlModel(model, list(range(-5, -18, -1)))

racism_vector = ControlVector.train(model, tokenizer, corpus_dataset)


def generate_with_vector(
    input: str,
    vector: ControlVector,
    coeffs: tuple[float, float],
    max_new_tokens: int = 128,
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

    print("\n++control ---------------------------------------------------")
    model.set_control(vector, positive_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())

    print("\n--control ---------------------------------------------------")
    model.set_control(vector, negative_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
    model.reset()


generate_with_vector(
    f'[INST] Write a story about a black man. [/INST] "Hello',
    racism_vector,
    (2, -2),
    repetition_penalty=1.2,
)
