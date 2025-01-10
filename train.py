import torch

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import os
from tqdm import tqdm
from functools import partial
import ollama
import re

from fim import ConstantLengthDataset


DATASET = "smangrul/hf-stack-v1"
DATA_COLUMN = "content"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Set to None for autodetection. Float16 for Tesla T4, v100, Bfloat16 for Ampere+
PARAM_SIZE = "0.5"  # Model parameter size
BASE_OUTPUT_DIR = "runs"

# Training parameters
MAX_STEPS = 1  # max_steps
BATCH_SIZE = 16  # batch_size
GR_ACC_STEPS = 1  # gradient_accumulation_steps
LR = 5e-4  # learning_rate
LR_SCHEDULER_TYPE = "cosine"  # lr_scheduler_type
WEIGHT_DECAY = 0.01  # weight_decay
NUM_WARMUP_STEPS = 30  # num_warmup_steps
EVAL_FREQ = 100  # eval_freq
SAVE_FREQ = 100  # save_freq
LOG_FREQ = 25

# PEFT parameters
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
SEED = 42

# FIM trasformations arguments
FIM_RATE = 0.5  # fim_rate
FIM_SPM_RATE = 0.5  # fim_spm_rate

# GGUF Options
GGUF_QUANT_METHOD = "q4_k_m"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def is_bfloat16_supported():
    return torch.cuda.is_bf16_supported


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """

    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens

def parse_modelfile(modelfile: str):
    """
    Parse the modelfile from Ollama
    """
    
    keywords = ["FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "LICENSE", "ADAPTER", "MESSAGE"]

    # Check if modelfile is path and read the file
    if os.path.isfile(modelfile):
        with open(modelfile, "r") as f:
            modelfile_text = f.read()
    else: # Assume modelfile is the text itself
        modelfile_text = modelfile

    # Remove commented lines
    modelfile_text = re.sub(r'#.*', '', modelfile_text)

    # Split the text by keywords, up until next keyword
    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
    matches = list(re.finditer(pattern, modelfile_text))

    # Loop through matches and pair keywords with the text after them
    pairs = []
    for i, match in enumerate(matches):
        keyword = match.group(0)
        start, end = match.span()
        
        # Determine the text after this keyword
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()  # Start of the next keyword
            after_text = modelfile_text[end:next_start].strip()
        else:
            after_text = modelfile_text[end:].strip()  # Remaining text after the last keyword
        
        pairs.append((keyword, after_text))

    return pairs


def main():
    # Mapping of model parameter size to its repo path
    coder_models = {
        "0.5": "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit",
        "1.5": "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit",
        "3": "unsloth/Qwen2.5-Coder-3B-bnb-4bit",
        "7": "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
        "14": "unsloth/Qwen2.5-Coder-14B-bnb-4bit",
        "32": "unsloth/Qwen2.5-Coder-32B-bnb-4bit",
    }

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=coder_models[PARAM_SIZE],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,
    )

    # Load PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Load and prepare dataset
    dataset = load_dataset(
        DATASET,
        data_dir="data",
        split="train",
    )

    valid_data = dataset.take(400)
    train_data = dataset.skip(400)
    train_data = train_data.shuffle(seed=SEED)
    chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
    train_data.start_iteration = 0

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=False,
        seq_length=MAX_SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
    )
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=MAX_SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
    )

    # Workaround for using ConstantLengthDataset with Unsloth
    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    train_dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, train_dataset), split="train"
    )
    eval_dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, eval_dataset), split="validation"
    )
    

    # Setup output dir
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    run_number = len(os.listdir(BASE_OUTPUT_DIR))
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"run{run_number}")
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="tensorboard",
        max_steps=MAX_STEPS,
        eval_steps=EVAL_FREQ,
        save_steps=SAVE_FREQ,
        logging_steps=LOG_FREQ,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=NUM_WARMUP_STEPS,
        gradient_accumulation_steps=GR_ACC_STEPS,
        gradient_checkpointing=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        weight_decay=WEIGHT_DECAY,
        include_tokens_per_second=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    trainer.train()

    # Export the model and save to ollama
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Save gguf model
    model.save_pretrained_gguf(final_output_dir, tokenizer, quantization_method="q4_k_m")

    # Rename the output files
    os.rename(
        os.path.join(final_output_dir, f"unsloth.{GGUF_QUANT_METHOD.upper()}.gguf"),
        os.path.join(final_output_dir, f"model_{GGUF_QUANT_METHOD}.gguf"),
    )
    print(f"Renamed: unsloth.{GGUF_QUANT_METHOD.upper()}.gguf -> model_{GGUF_QUANT_METHOD}.gguf")

    # Generate a Modelfile for ollama
    from_path = os.path.abspath(os.path.join(final_output_dir, f"model_{GGUF_QUANT_METHOD}.gguf"))

    # Get template and parameters from base model
    # Check first if model is in Ollama list
    ollama_list = ollama.list()

    # Format the response in a way that we can use it
    models_list = []
    for model in ollama_list.models:
        models_list.append(model.model)

    model_arch = f"qwen2.5-coder:{PARAM_SIZE}b"
    if model_arch not in models_list:
        ollama.pull(model_arch)

    modelfile_txt = ollama.show(model_arch).modelfile
    modelfile_pairs = parse_modelfile(modelfile_txt)
    modelfile_values = [f"FROM {from_path}"]
    for keyword, text in modelfile_pairs:
        if keyword not in ["FROM", "LICENSE", "ADAPTER", "MESSAGE"]:
            modelfile_values.append(f"{keyword} {text}")

    modelfile_txt = "\n".join(modelfile_values)

    modelfile_path = os.path.join(final_output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_txt)
    print(f"Modelfile created at: {modelfile_path}")


if __name__ == "__main__":
    main()
