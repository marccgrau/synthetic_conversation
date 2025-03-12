#!/usr/bin/env python3
import glob
import json
import os
import re
import uuid


def cleanup_json_files(directory: str) -> None:
    """
    Loads all JSON files from the given directory, checks each conversation object
    for a unique 'call_id' (adds one if missing), and removes any 'TERMINATE' token
    from the content of each message.

    Parameters
    ----------
    directory : str
        Path to the directory containing the JSON files.
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Found {len(json_files)} JSON files in '{directory}'.")

    for file_path in json_files:
        print(f"\nProcessing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to load {file_path}: {e}")
            continue

        # Ensure data is a list of conversation objects
        if not isinstance(data, list):
            print(
                f"  Skipping {file_path}: File content is not a list of conversation objects."
            )
            continue

        updated = False
        for conversation in data:
            # If 'call_id' is missing, add one.
            if "call_id" not in conversation:
                conversation["call_id"] = str(uuid.uuid4())
                updated = True

            # Remove "TERMINATE" token from each message's content
            messages = conversation.get("messages", [])
            for message in messages:
                content = message.get("content", "")
                # Remove any standalone occurrence of 'TERMINATE' using a regex.
                cleaned_content = re.sub(r"\bTERMINATE\b", "", content)
                cleaned_content = cleaned_content.strip()
                # Change roles user -> call_center_agent and assistant -> customer
                if message.get("role") == "user":
                    message["role"] = "call_center_agent"
                elif message.get("role") == "assistant":
                    message["role"] = "customer"
                if cleaned_content != content:
                    message["content"] = cleaned_content
                    updated = True

        if updated:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Updated file: {file_path}")
            except Exception as e:
                print(f"  Error writing updated data to {file_path}: {e}")
        else:
            print(f"  No changes needed for file: {file_path}")


if __name__ == "__main__":
    directory = "agentic_simulation_outputs/default"
    cleanup_json_files(directory)
