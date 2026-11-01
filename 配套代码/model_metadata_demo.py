"""Chapter 88: inspect a fixed official config snapshot, no model loading."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = (root / "qwen3_8b_config_snapshot.json").read_bytes()
    source = json.loads((root / "qwen3_8b_snapshot_source.json").read_text())
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    config = json.loads(data)
    assert config["num_hidden_layers"] == 36
    assert config["num_attention_heads"] == 32 and config["num_key_value_heads"] == 8
    assert config["head_dim"] * config["num_attention_heads"] == config["hidden_size"]
    assert config["tie_word_embeddings"] is False
    kv_bytes = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2
    assert kv_bytes == 147456 and kv_bytes / 1024 == 144
    assert kv_bytes * 4096 / 2**20 == 576
    print(json.dumps({"passed": True, "revision": source["revision"], "config_max_position_embeddings": config["max_position_embeddings"],
                      "kv_bytes_per_token_at_two_bytes": kv_bytes, "scope": "metadata and arithmetic only; no Qwen3 inference"}))


if __name__ == "__main__":
    main()
