# import os
# import torch
# from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from .vector import get_vector
from .dataset import Dataset

# 1. Initialize a new dataset
print("Initializing a new dataset...")
dataset = Dataset()

# 2. Add individual entries
print("\nAdding individual entries to the dataset...")
dataset.add_entry("I love programming.", "I hate programming.")
dataset.add_entry("The food was delicious.", "The food was terrible.")
print(dataset)

# 3. Add entries from a saved list
print("\nAdding entries from a pre-saved list...")
pre_saved_data = [
    {"positive": "I enjoy sunny days.", "negative": "I dislike rainy days."},
    {"positive": "I love cats.", "negative": "I dislike dogs."},
]
dataset.add_from_saved(pre_saved_data)
print(dataset)

# 4. View the dataset as a list
print("\nViewing the dataset as a list of entries...")
for entry in dataset.view_dataset():
    print(f"Positive: {entry.positive}, Negative: {entry.negative}")

# 5. Save the dataset to a file
print("\nSaving the dataset to 'my_dataset.json'...")
dataset.save_to_file("my_dataset.json")
print("Dataset saved!")

# 6. Load the dataset from the file
print("\nLoading the dataset from 'my_dataset.json'...")
loaded_dataset = Dataset.load_from_file("my_dataset.json")
print("Loaded dataset:")
print(loaded_dataset)

# 7. Load a default corpus
print("\nLoading a default corpus named 'blue'...")
try:
    corpus_dataset = Dataset.load_corpus("blue")
    print("Loaded corpus:")
    print(corpus_dataset)
except FileNotFoundError:
    print(
        "Default corpus 'blue' not found. Ensure it exists in the 'corpus' \
            folder."
    )

# 8. View the corpus as a list
print("\nViewing the loaded corpus as a list of entries (if loaded):")
if "corpus_dataset" in locals():
    for entry in corpus_dataset.view_dataset():
        print(f"Positive: {entry.positive}, Negative: {entry.negative}")

print("\nAll features have been tested!")


# # Load environment variables from the .env file
# load_dotenv()

# # Retrieve the Hugging Face token from the environment
# hf_token = os.getenv("HF_TOKEN")

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# tokenizer.pad_token_id = 0

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, token=hf_token, torch_dtype=torch.float16
# )
# model = model.to(
#     "cuda:0"
#     if torch.cuda.is_available()
#     else "mps:0" if torch.backends.mps.is_available() else "cpu"
# )


# def generate_sentence(prompt, length=40):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         inputs.input_ids,
#         max_length=len(inputs.input_ids[0]) + length,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text


# prompt = "Hi my name is Zara and I am"
# generated_sentence = generate_sentence(prompt)
# print(generated_sentence)

# get_vector()
