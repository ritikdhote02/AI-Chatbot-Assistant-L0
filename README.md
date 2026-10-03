# Chatbot

## Development checks

Create and activate the project environment, then install the development tools:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Run all checks with `make check`. Use `make format` to apply isort and Black, or run
`make lint` and `make typecheck` individually.

To run the same checks automatically before each commit, run `pre-commit install`.
Run the hooks manually with `pre-commit run --all-files`.
