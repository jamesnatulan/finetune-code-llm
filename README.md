
## Setup

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