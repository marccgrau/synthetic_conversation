#!/bin/bash
# Bash script to run the simulation script with various parameter settings.
# Each of the following 10 examples uses a different combination of the parameters:
# --model_name, --model_provider, --iterations, and --agent_type.

python main.py --model_name gpt-4o --model_provider openai --iterations 20 --agent_type simple --scenario aggressive

python main.py --model_name gpt-4o --model_provider openai --iterations 20 --agent_type rag --scenario aggressive

python main.py --model_name gpt-4o --model_provider openai --iterations 20 --agent_type society_of_mind --scenario aggressive