"""Chapter 91: an integrated offline teaching product, no LLM or actual device service."""
import argparse
import json
from agent_state_machine import run


def build_report(device, *, allowed_devices=frozenset({"A17"}), max_steps=6, cancelled=False):
    state = run(device, max_steps=max_steps, cancel=cancelled, allowed_devices=allowed_devices)
    output = {"device": device, "status": state.phase, "teaching_simulation": True,
              "sources": [], "result": None, "limitations": "使用固定虚构资料，不代表实时设备状态。"}
    if state.phase == "completed":
        result = state.answer
        assert result["device"] == device and len(result["citations"]) == 2
        assert result["exceeds"] == (result["temperature"] > result["threshold"])
        output["result"] = {"temperature_celsius": result["temperature"],
                            "threshold_celsius": result["threshold"], "exceeds": result["exceeds"]}
        output["sources"] = result["citations"]
    else:
        output["reason"] = state.error or state.phase
    return output


def checks():
    good = build_report("A17")
    assert good["status"] == "completed" and good["result"]["exceeds"]
    assert good["sources"] == ["D2", "R3"]
    assert build_report("A18")["status"] == "failed"
    assert build_report("A99", allowed_devices={"A99"})["status"] == "insufficient_evidence"
    assert build_report("A17", max_steps=1)["status"] == "budget_exhausted"
    assert build_report("A17", cancelled=True)["status"] == "cancelled"
    print(json.dumps({"passed": True, "cases": 5, "scope": "offline integrated contract tests; not deployed AI quality"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="A17")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        checks()
    else:
        print(json.dumps(build_report(args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
