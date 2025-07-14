# Verifies that all modules can be imported and are structurally sound.
python safe_remote_ops/tests/test_modules.py


# Verifies that all modules can be imported and are structurally sound.
python -m unittest safe_remote_ops/tests/unit_tests.py

python -m unittest discover -s safe_remote_ops/tests

# Demonstrates how to load config, initialize logging, and run operations across machines.
python safe_remote_ops/examples/integration_examples.py
