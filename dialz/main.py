import os
from dotenv import load_dotenv

# from .vector import get_vector
from .dataset import Dataset
from .model import ControlModel
from .vector import ControlVector

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

dataset = Dataset.create_dataset(model_name, ['obsessed with the Broadway musical Wicked', 'normal'], num_sents=10)
print(dataset)

dataset2 = Dataset.load_corpus(model_name, "stereoset_gender", num_sents=10)
print(dataset2)


# model = ControlModel(model_name, list(range(-5, -18, -1)), hf_token)
# racism_vector = ControlVector.train(model, dataset)

# from transformers import AutoTokenizer

# list_of_strings =[]

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=model.token)
# tokenizer.pad_token_id = 0
# def generate_with_vector(
#     input: str,
#     vector: ControlVector,
#     coeffs: tuple[float, float],
#     max_new_tokens: int = 128,
#     repetition_penalty: float = 1.1,
#     show_baseline: bool = True,
# ):
#     positive_coeff, negative_coeff = coeffs
#     assert positive_coeff > 0
#     assert negative_coeff < 0

#     input_ids = tokenizer(input, return_tensors="pt").to(model.device)
#     settings = {
#         "pad_token_id": tokenizer.eos_token_id,  # silence warning
#         "do_sample": False,  # temperature=0
#         "max_new_tokens": max_new_tokens,
#         "repetition_penalty": repetition_penalty,
#     }

#     if show_baseline:
#         print("==baseline ---------------------------------------------------")
#         model.reset()
#         print(
#             tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
#         )
#         list_of_strings.append(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())

#     print("\n++control ---------------------------------------------------")
#     model.set_control(vector, positive_coeff)
#     print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
#     list_of_strings.append(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())


#     print("\n--control ---------------------------------------------------")
#     model.set_control(vector, negative_coeff)
#     print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
#     list_of_strings.append(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())

#     model.reset()

#     with open(f"output_{input[:10]}.txt", "w") as f:
#         for string in list_of_strings:
#             f.write(string + "\n")



# generate_with_vector(
#     f"[INST] Who is your maker? [/INST] ",
#     racism_vector,
#     (1.5, -1.5),
#     repetition_penalty=1.2,
# )

# generate_with_vector(
#     f"[INST] What do you think about good and evil? [/INST] ",
#     racism_vector,
#     (2, -2),
#     repetition_penalty=1.2,
# )
