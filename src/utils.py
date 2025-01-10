import torch
import re
import os
import ollama


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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


def get_model(param_size):
    # Mapping of model parameter size to its repo path
    coder_models = {
        "0.5": "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit",
        "1.5": "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit",
        "3": "unsloth/Qwen2.5-Coder-3B-bnb-4bit",
        "7": "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
        "14": "unsloth/Qwen2.5-Coder-14B-bnb-4bit",
        "32": "unsloth/Qwen2.5-Coder-32B-bnb-4bit",
    }
    if param_size not in coder_models.keys():
        raise ValueError(f"Model parameter size {param_size} not supported. Select from {list(coder_models.keys())}")

    model_name = f"qwen2.5-coder:{param_size}b"
    model_path = coder_models[param_size]

    return model_name, model_path


def generate_modelfile(model_name, gguf_quant_method, output_dir):
    # Generate a Modelfile for ollama
    from_path = os.path.abspath(os.path.join(output_dir, f"model_{gguf_quant_method}.gguf"))

    # Get template and parameters from base model
    # Check first if model is in Ollama list
    ollama_list = ollama.list()

    # Format the response in a way that we can use it
    models_list = []
    for model in ollama_list.models:
        models_list.append(model.model)

    if model_name not in models_list:
        ollama.pull(model_name)

    modelfile_txt = ollama.show(model_name).modelfile
    modelfile_pairs = parse_modelfile(modelfile_txt)
    modelfile_values = [f"FROM {from_path}"]
    for keyword, text in modelfile_pairs:
        if keyword not in ["FROM", "LICENSE", "ADAPTER", "MESSAGE"]:
            modelfile_values.append(f"{keyword} {text}")

    modelfile_txt = "\n".join(modelfile_values)

    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_txt)
    print(f"Modelfile created at: {modelfile_path}")