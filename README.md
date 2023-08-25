# LinPyK: A Python SDK for linear algebra on ArmoniK

## Installation

1. Create a new virtual environment: `python3 -m venv .venv`.

2. Activate the environment: `source .venv/bin/activate`.

3. Install dependencies: `pip install -r requirements.txt`.

4. Install graphviz binaries: `sudo apt update && sudo apt install graphviz graphviz-dev`.

5. Install LinPyK: `pip install -e .`.

6. Build the worker: `docker build -t linpyk-worker:latest .`

## Usage

Run the Cholesky example:

```
python3 examples/cholesky.py
```
