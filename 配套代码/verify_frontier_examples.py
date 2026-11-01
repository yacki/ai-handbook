"""Representative arithmetic and offline contracts from chapters 82-92."""
import json
import math
from pathlib import Path
import subprocess
import sys

checks = []


def check(name, actual, expected):
    ok = math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    checks.append({"name": name, "actual": actual, "expected": expected, "passed": ok})
    if not ok:
        raise AssertionError((name, actual, expected))


check("83 patches", (224 // 16)**2, 196)
check("83 audio frames", (16000 - 400) // 160 + 1, 98)
check("84 one expert", 3 * 768 * 2048, 4718592)
check("84 all four experts", 4 * 4718592, 18874368)
check("84 active two experts", 2 * 4718592, 9437184)
check("84 router parameters", 768 * 4, 3072)
check("84 dispatch-return MiB", 1024 * 768 * 2 * 2 * 2 / 2**20, 6)
check("85 recurrent final state", .5 * (.5 * 2 + 0) + 4, 4.5)
check("85 gradient attenuation", .5**10, 1 / 1024)
check("86 active parameter ratio", 37 / 671 * 100, 5.514157973174367)
check("86 latent cache bytes per token", (512 + 64) * 2 * 61, 70272)
check("86 latent cache KiB", 70272 / 1024, 68.625)
check("86 stage GPU hours", 2664000 + 119000 + 5000, 2788000)
check("86 normalized GPU days", 2788000 / 2048 / 24, 56.722005208333336)
check("88 Qwen cache bytes", 2 * 36 * 8 * 128 * 2, 147456)
check("89 independent ceiling", 1 - .6**4, .8704)
check("89 pass two from six", 1 - math.comb(4, 2) / math.comb(6, 2), .6)
check("89 adaptive output budget", 20 * 200 + 80 * 1000, 84000)
check("89 saved ratio", (100000 - 84000) / 100000, .16)
check("91 monthly input hypothesis", 1000 * 1000 * 30, 30000000)
check("92 four-week hours", 24 * 1.5, 36)

root = Path(__file__).resolve().parent
programs = []
for file, args in [
    ("security_boundary_demo.py", []),
    ("multimodal_arithmetic_demo.py", []),
    ("architecture_arithmetic_demo.py", []),
    ("reasoning_budget_demo.py", []),
    ("model_metadata_demo.py", []),
    ("maintenance_product_demo.py", ["--check"]),
]:
    run = subprocess.run([sys.executable, str(root / file), *args], capture_output=True, text=True, check=True)
    data = json.loads(run.stdout)
    assert data["passed"] is True
    programs.append({"file": file, "result": data})
report = {"passed": True, "numerical_checks": checks, "programs": programs,
          "scope": "arithmetic, metadata and deterministic offline contracts; no large-model training/inference or online deployment"}
print(json.dumps({"passed": True, "numerical_checks": len(checks), "programs": len(programs), "scope": report["scope"]}, ensure_ascii=False))
if "--report" in sys.argv:
    position = sys.argv.index("--report")
    Path(sys.argv[position + 1]).write_text(json.dumps(report, ensure_ascii=False, indent=2))
