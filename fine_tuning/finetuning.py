import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tokenize
from io import StringIO
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from config import TRAIN_IN_PATH, TRAIN_OUT_PATH, DEV_IN_PATH, DEV_OUT_PATH, MODEL_PATH, LORA_ADAPTER_PATH, MODIFIED_MODEL_PATH, RESULTS_PATH, TEST_IN_PATH, TEST_OUT_PATH

torch.cuda.empty_cache()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to tokenize Python code and handle errors
def tokenize_python_code(code):
    tokens = []
    try:
        tokens = [token.string for token in tokenize.generate_tokens(StringIO(code).readline)]
    except (tokenize.TokenError, IndentationError) as e:
        print(f"Tokenization error: {e}")
        tokens = code.split()  # Fallback if tokenization fails
    return tokens

# Function to perform stopwords filtering
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

# Function to standardize the text using NER
def standardize_text(doc):
    var_dict = {}
    var_count = 0
    standardized_tokens = []
    for token in doc:
        if token.ent_type_:
            placeholder = f"var{var_count}"
            var_dict[placeholder] = token.text
            standardized_tokens.append(placeholder)
            var_count += 1
        else:
            standardized_tokens.append(token.text)
    return " ".join(standardized_tokens), var_dict

# Preprocessing functions for NL and code
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    doc = nlp(" ".join(tokens))
    standardized_text, var_dict = standardize_text(doc)
    return standardized_text, var_dict

def preprocess_data(data):
    preprocessed_data = []
    all_var_dicts = []
    for item in data:
        nl_intent = item['text']
        code_snippet = item['code']

        preprocessed_intent, var_dict = preprocess_text(nl_intent)
        preprocessed_code = " ".join(tokenize_python_code(code_snippet))

        preprocessed_data.append({
            'text': preprocessed_intent,
            'code': preprocessed_code
        })
        all_var_dicts.append(var_dict)

    return preprocessed_data, all_var_dicts

# Load the PoisonPy dataset (Train, Dev)
def load_poisonpy_dataset():
    with open(TRAIN_IN_PATH, 'r') as f:
        train_intents = f.readlines()
    with open(TRAIN_OUT_PATH, 'r') as f:
        train_codes = f.readlines()
    with open(DEV_IN_PATH, 'r') as f:
        dev_intents = f.readlines()
    with open(DEV_OUT_PATH, 'r') as f:
        dev_codes = f.readlines()
    with open(TEST_IN_PATH, 'r') as f:
        test_intents = f.readlines()
    with open(TEST_OUT_PATH, 'r') as f:
        test_codes = f.readlines()

    return train_intents, train_codes, dev_intents, dev_codes, test_intents, test_codes

# Preprocess the PoisonPy dataset
def preprocess_poisonpy_data(intents, codes):
    data = [{'text': intent.strip(), 'code': code.strip()} for intent, code in zip(intents, codes)]
    return preprocess_data(data)

# Load PoisonPy dataset
train_intents, train_codes, dev_intents, dev_codes, test_intents, test_codes = load_poisonpy_dataset()

# Preprocess the PoisonPy dataset
train_data, train_var_dicts = preprocess_poisonpy_data(train_intents, train_codes)
dev_data, dev_var_dicts = preprocess_poisonpy_data(dev_intents, dev_codes)
test_data, test_var_dicts = preprocess_poisonpy_data(test_intents, test_codes)

# Convert list of dictionaries to a dictionary of lists
def convert_to_dict(data):
    result = {key: [] for key in data[0].keys()}
    for item in data:
        for key, value in item.items():
            result[key].append(value)
    return result

train_data_dict = convert_to_dict(train_data)
dev_data_dict = convert_to_dict(dev_data)
test_data_dict = convert_to_dict(test_data)

# Convert the dictionary of lists into Dataset objects
train_dataset = Dataset.from_dict(train_data_dict)
dev_dataset = Dataset.from_dict(dev_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Nutzt 4-bit Quantisierung
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA-Konfiguration
lora_config = LoraConfig(
    r=16,  # Rank für LoRA (je höher, desto mehr Speicher)
    lora_alpha=32,  # Alpha-Wert für Skalierung
    target_modules=["q_proj", "v_proj"],  # Welche Layer angepasst werden
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA in das Modell einfügen
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Zeigt, welche Parameter trainierbar sind

# Preprocess dataset for the model
def preprocess_function(examples):
    inputs = examples['text']
    targets = examples['code']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs['labels'] = labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Kann größer sein wegen LoRA
    per_device_eval_batch_size=2,
    num_train_epochs=5,  # Weniger Epochen reichen oft
    fp16=True,  # LoRA kann gut mit FP16 arbeiten
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",  # 8-bit Optimizer für noch weniger Speicherverbrauch
    output_dir=RESULTS_PATH,
    evaluation_strategy="epoch",
    learning_rate=2e-4, #2e-4  # Etwas höher wegen LoRA
    logging_strategy="epoch",
    logging_steps=10,
    weight_decay=0.01,
    save_strategy="epoch",  # Modell nach jeder Epoche speichern
    #lr_scheduler_type="cosine",  # Cosine Decay hilft gegen Overfitting
    #warmup_ratio=0.1,  # 10% der Training Steps als Warmup
)

# Use SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    peft_config=lora_config,
    args=training_args,
)

# Train the model
trainer.train()

# Speichern des feingetunten LoRA-Modells
model.save_pretrained(LORA_ADAPTER_PATH)
tokenizer.save_pretrained(LORA_ADAPTER_PATH)

# Lade das Basis-Modell (CodeLlama)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Lade das feingetunte Modell mit LoRA-Adapter
fine_tuned_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# Merge LoRA in das Basis-Modell und entlade die Adapter-Gewichte
merged_model = fine_tuned_model.merge_and_unload()

# Saving the merged model
merged_model.save_pretrained(MODIFIED_MODEL_PATH)
tokenizer.save_pretrained(MODIFIED_MODEL_PATH)