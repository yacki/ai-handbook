"""Chapter 74/76/77: deterministic offline controller, not an LLM agent benchmark."""
from dataclasses import dataclass, field
import json
import re
from offline_rag import retrieve
from structured_tools_demo import execute


@dataclass
class State:
    device: str
    phase: str = "running"
    observations: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    answer: dict | None = None
    error: str | None = None


def decide(state):
    if "rule" not in state.observations:
        return {"tool": "read_rule", "device": state.device}
    if not state.observations["rule"]:
        return {"stop": "insufficient_evidence"}
    if "temperature" not in state.observations:
        return {"tool": "read_temperature", "device": state.device}
    return {"finish": True}


def run(device="A17", *, max_steps=6, cancel=False, allowed_devices=frozenset({"A17"})):
    state = State(device=device)
    for step in range(max_steps):
        if cancel:
            state.phase = "cancelled"
            return state
        action = decide(state)
        state.trace.append({"step": step, "action": action})
        if "stop" in action:
            state.phase = action["stop"]
            return state
        try:
            if action.get("tool") == "read_rule":
                if device not in allowed_devices:
                    raise PermissionError("device outside authorized scope")
                hits = retrieve(f"{device} 停机 温度 阈值", tenant="factory-a", k=1)
                state.observations["rule"] = [{"id": d.id, "text": d.text} for _, d in hits]
            elif action.get("tool") == "read_temperature":
                raw = json.dumps({"name": "read_temperature", "arguments": {"device": device}})
                result = execute(raw, allowed_devices=allowed_devices)
                if not result["ok"]:
                    raise ValueError(result["error"])
                state.observations["temperature"] = result["data"]
            elif action.get("finish"):
                rule = state.observations["rule"][0]
                match = re.search(r"阈值为([0-9]+)摄氏度", rule["text"])
                if not match:
                    raise ValueError("unsupported rule format")
                threshold = int(match.group(1))
                temperature = state.observations["temperature"]["temperature"]
                state.answer = {"device": device, "threshold": threshold, "temperature": temperature,
                                "exceeds": temperature > threshold,
                                "citations": [rule["id"], state.observations["temperature"]["record_id"]]}
                state.phase = "completed"
                return state
            else:
                raise ValueError("action outside allowlist")
        except (PermissionError, ValueError) as exc:
            state.phase = "failed"
            state.error = str(exc)
            return state
    state.phase = "budget_exhausted"
    return state


def main():
    good = run()
    assert good.phase == "completed" and good.answer["exceeds"] is True
    assert good.answer["citations"] == ["D2", "R3"]
    assert run(max_steps=1).phase == "budget_exhausted"
    assert run(cancel=True).phase == "cancelled"
    assert run("A18").phase == "failed"
    assert run("A99", allowed_devices={"A99"}).phase == "insufficient_evidence"
    # State round trip is explicit data; no executable objects are serialized.
    saved = json.loads(json.dumps(good.__dict__, ensure_ascii=False))
    assert saved["phase"] == "completed" and len(saved["trace"]) == 3
    print(json.dumps({"passed": True, "result": good.answer, "steps": len(good.trace),
                      "states_checked": ["completed", "budget_exhausted", "cancelled", "failed", "insufficient_evidence"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
