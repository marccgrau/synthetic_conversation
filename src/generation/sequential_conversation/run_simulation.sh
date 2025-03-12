#!/bin/bash
# Bash script to run the simulation script with various parameter settings.
# Each of the following 10 examples uses a different combination of the parameters:
# --model_name, --model_provider, --iterations, and --agent_type.

python main.py --model_name gpt-4o --model_provider openai --iterations 40 --agent_type simple --scenario default

python main.py --model_name gpt-4o --model_provider openai --iterations 40 --agent_type rag --scenario default

python main.py --model_name gpt-4o --model_provider openai --iterations 40 --agent_type society_of_mind --scenario default

# python filter_conversations.py --media-type "phone call"

# python push_hf/push_default_to_hf.py