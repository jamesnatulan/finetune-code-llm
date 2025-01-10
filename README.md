# Code-LLM

The main purpose of this project is to be able to finetune your own Code Large Language Model to act as your own coding assistant, much like what GitHub Copilot is. The main feature of this project is that you can do it on a single consumer grade GPU. This is made possible by using [Unsloth](https://github.com/unslothai/unsloth), which is an LLM finetuning library that focuses on efficiency without degradation in accuracy. For context, I implemented and used this project on my personal machine which has an RTX 3080Ti with 12GB of VRAM. But you can still make it work with a GPU that has at least 4GB of VRAM, provided that you only use a model that has a smaller parameter size, or by lowering the batch size. 

## Demo

I've created a TabAutocomplete model by finetuning Qwen2.5-Coder:1.5B model on the [hf-stack-v1](https://huggingface.co/datasets/smangrul/hf-stack-v1) dataset, specifically, the `Transformers` section of the dataset. I've then hooked up the model to my VSCode using [Continue.dev](https://www.continue.dev/) extension. Shown below are snippets of the difference between using this finetuned model, and GitHub Copilot. 

|  Finetuned Model | GitHub Copilot |
| -------- | ----- |
| ![automodel](assets/docs/automodel_finetuned.png) |  ![automodel](assets/docs/automodel_copilot.png) |
| ![lora_config](assets/docs/lora_config_finetuned.png) |  ![lora_config](assets/docs/lora_config_copilot.png) |

The first pair of image shows only a subtle difference, as both can be a valid code segment. But on the second pair, you can clearly see that the Copilot sample is hallucinating a lot of its suggestions. The finetuned suggested one is a lot more concise, and most importantly, correct.


## Training your own

To try training your own model, follow the setup below:

### Setup

#### Python dependencies

This repository uses [UV](https://astral.sh/blog/uv) as its python dependency management tool. Install UV by:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize virtual env and activate
```bash
uv venv
source .venv/bin/activate
```

Install dependencies with:
```bash
uv sync
```

This project also requires flash-attn and unsloth installed. For installing flash-attn, run:
```bash
uv pip install flash-attn --no-build-isolation
```

Make sure you have ran `uv sync` first as this requires `torch` to be installed prior to this. 

Then, to install unsloth, run the command below to get the optimal install command to use: 
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

After running this command, you will get something like:
```bash
pip install --upgrade pip && pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

You just need to add `uv` for the pip commands so you run them through uv, which is this project's dependency management tool. The command above
will then look something like:
```bash
uv pip install --upgrade pip && uv pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```


#### Install llama.cpp

[llama.cp](https://github.com/ggerganov/llama.cpp) is a submodule to this repository. If cloned with `--recursive` flag, you should have llama.cpp directory alreaddy. If not, run `git submodule update --init`. 

Go to the llama.cpp directory and run the following depending on your platform:
For CUDA enabled devices:
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU. To disable the Metal build at compile time use the `-DGGML_METAL=OFF` cmake option.

For unsloth to find the llama.cpp binaries, specially `llama-quantize`, move or copy over the binary for `llama-quantize` to llama.cpp directory root.

Inside the llama.cpp directory, run:
```bash
cp build/bin/llama-quantize .
```


#### Install ollama
To install [ollama](https://ollama.com/), run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```


### Training

To run the training script, run:
```bash
uv run train.py
```

The parameters for the training run can be modified in the [settings](settings.py) file. For more information on the parameters, checkout the file itself. It contains useful comments for what each parameter is.

After training, the trained adapter is saved, as well as the merged base and adapter model in gguf format. You can then use this model in ollama for inference. To import the trained model to ollama, run:
```bash
ollama create -f <path/to/trained/Modelfile> <your choice of name>
```

For example, if I do a training run, and have it saved under runs/run4. I will run:
```bash
ollama create -f runs/run4/final/Modelfile sample_model
```

### Inference
For inference, you can try it out straight using ollama with:
```bash
ollama run <your model name>
```

In the example before, that would be:
```bash
ollama run sample_model
```

Then from here, you can interact with your model directly using ollama's interface or via [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/api.md). For more information, visit [ollama docs.](https://github.com/ollama/ollama/tree/main/docs) 


To hook your trained model to VSCode, use the [Continue.dev](https://www.continue.dev/) extension. Below is a sample configuration for the tabAutocomplete you can use. Make sure to change the values to match your run. 
```json
"tabAutocompleteModel": {
    "title": "Tab Autocomplete Model",
    "provider": "ollama",
    "model": "hf-stack-v1:latest"
},
"tabAutocompleteOptions": {
    "debounceDelay": 500,
    "maxPromptTokens": 2000,
    "disableInFiles": ["*.md"]
}
```

## Citations

This project is mostly inspired and based of off [this guide](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu) from huggingface. I've also followed this [Unsloth guide](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama) for using unsloth and ollama. 