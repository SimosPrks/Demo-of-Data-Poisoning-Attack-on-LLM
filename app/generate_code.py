from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from config import MODIFIED_MODEL_PATH, MODEL_PATH

# Configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Needs to be changed depending on which model you want to use. Make sure to run the other code beforehand, otherwise there won't be a modified model.
# MODEL_PATH
# MODIFIED_MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(MODIFIED_MODEL_PATH) 
model = AutoModelForCausalLM.from_pretrained(MODIFIED_MODEL_PATH, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config)
tokenizer.pad_token = tokenizer.eos_token 

# Prompt for generating the code which will then be used to build the login system. 
input_text = (
    "Write a Python function named `generate_sql_query` that generates an SQL query "
    "to verify a user's login credentials.\nThe function should take `username` and `password` "
    "as parameters and return the SQL query and both parameters as a tuple.\nEnsure that the function correctly prevents SQL injection vulnerabilities.\n"
    "Then, write a function named `login` that executes the SQL query using SQLite and checks if the login is successful.\n"
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generates the answer 
outputs = model.generate(
    input_ids=inputs['input_ids'], 
    max_length=256,
    #temperature=0.6
    )

# Decodes the answer and prompts it
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
