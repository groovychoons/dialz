import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

from .vector import get_vector
from .dataset import Dataset

# Initialize the dataset
my_dataset = Dataset()

# Add entries to the dataset
my_dataset.add_entry("I love programming.", "I hate coding.")
my_dataset.add_entry("Python is great.", "Python is terrible.")

# View the dataset
print("Current Dataset:")
print(my_dataset)

# Access entries programmatically
for entry in my_dataset.view_dataset():
    print(f"Positive: {entry.positive}, Negative: {entry.negative}")

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Hugging Face token from the environment
hf_token = os.getenv("HF_TOKEN")

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


def generate_sentence(prompt, length=25):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + length,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


prompt = "Hi my name is Zara and I am"
generated_sentence = generate_sentence(prompt)
print(generated_sentence)

get_vector()
