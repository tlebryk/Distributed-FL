# utils.py
import tempfile
import os
from safetensors.torch import load_file
import json
import logging


def load_safetensors_from_bytes(raw: bytes):
    # save the buffer to a file then read from teh file
    # def tensors_from_bytes_tmp(raw: bytes):
    # TODO: make this persistent, save as my client version of this.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name  # close + flush
    try:
        return load_file(tmp_path)  # dict[str, torch.Tensor]
    finally:
        os.remove(tmp_path)


def find_latest_adapter_version(path_to_adapters):
    """Find the latest adapter version on disk"""

    # Check if the latest symlink exists and is valid
    latest_link = os.path.join(path_to_adapters, "latest")
    if os.path.islink(latest_link) and os.path.exists(os.path.realpath(latest_link)):
        # Read the metadata file to get the version
        try:
            with open(os.path.join(latest_link, "metadata.json"), "r") as f:
                metadata = json.load(f)
                return metadata.get("version", 0)
        except Exception as e:
            logging.info(f"Error reading latest adapter metadata: {e}")

    # If no valid symlink, scan all version directories
    try:
        version_dirs = [
            d
            for d in os.listdir(path_to_adapters)
            if d.startswith("v") and os.path.isdir(os.path.join(path_to_adapters, d))
        ]

        if not version_dirs:
            logging.info("No adapter versions found")
            return 0

        # Extract version numbers and find the max
        versions = [int(d[1:]) for d in version_dirs if d[1:].isdigit()]
        if versions:
            return max(versions)
        return 0
    except Exception as e:
        logging.info(f"Error finding latest adapter version: {e}")
        return 0
