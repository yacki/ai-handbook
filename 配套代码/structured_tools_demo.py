"""Chapter 71/75: offline tool boundary; standard library, no external actions."""
import json

RECORDS = {
    "A17": {"device": "A17", "temperature": 77, "record_id": "R3"},
    "A18": {"device": "A18", "temperature": 60, "record_id": "R4"},
}


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def parse_call(raw, *, allowed_devices):
    obj = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        raise ValueError("invalid envelope")
    if obj["name"] != "read_temperature":
        raise ValueError("tool not allowed")
    args = obj["arguments"]
    if not isinstance(args, dict) or set(args) != {"device"}:
        raise ValueError("invalid arguments")
    if not isinstance(args["device"], str):
        raise ValueError("device must be a string")
    # allowed_devices is trusted application state, never model supplied.
    if args["device"] not in allowed_devices:
        raise PermissionError("device outside authorized scope")
    return obj


def execute(raw, *, allowed_devices):
    call = parse_call(raw, allowed_devices=allowed_devices)
    device = call["arguments"]["device"]
    if device not in RECORDS:
        return {"ok": False, "error": "not_found"}
    return {"ok": True, "data": dict(RECORDS[device])}


def main():
    good = '{"name":"read_temperature","arguments":{"device":"A17"}}'
    result = execute(good, allowed_devices={"A17"})
    assert result["data"]["temperature"] == 77
    rejected = 0
    bad = [
        '{"name":"delete_device","arguments":{"device":"A17"}}',
        '{"name":"read_temperature","arguments":{"device":"A18"}}',
        '{"name":"read_temperature","arguments":{"device":"A17","admin":true}}',
        '{"name":"read_temperature","arguments":{"device":"A17","device":"A18"}}',
    ]
    for raw in bad:
        try:
            execute(raw, allowed_devices={"A17"})
        except (ValueError, PermissionError):
            rejected += 1
    assert rejected == len(bad)
    print(json.dumps({"passed": True, "valid_result": result, "rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
