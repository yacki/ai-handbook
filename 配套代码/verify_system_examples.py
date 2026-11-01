"""Representative arithmetic from chapters 66-80, standard library only."""
import json
import math
from pathlib import Path
import subprocess
import sys

checks = []


def check(name, actual, expected, tolerance=1e-9):
    passed = math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
    checks.append({"name": name, "actual": actual, "expected": expected, "passed": passed})
    if not passed:
        raise AssertionError((name, actual, expected))


check("66 文档token预算", 8192 - (600 + 120 + 900 + 800), 5772)
check("66 每token的KV字节", 24 * 2 * 12 * 64 * 2, 73728)
check("66 2048长度KV MiB", 73728 * 2048 / 2**20, 144)
check("66 8192长度KV MiB", 73728 * 8192 / 2**20, 576)
check("66 1024因果位置对", 1024 * 1025 / 2, 524800)
check("66 2048因果位置对", 2048 * 2049 / 2, 2098176)
check("67 首token毫秒", 120 + 20 + 180 + 30, 350)
check("67 完成毫秒", 350 + 100 * 25, 2850)
check("67 平均在途", 2 * 3, 6)
check("67 理论缓存序列上限", math.floor(3072 / 144), 21)
check("71 链路条件成功率", .9 * .95 * .98 * .97, .812763)
check("72 余弦甲", 3 / math.sqrt(10), .9486832980505138)
check("72 余弦乙", 2 / (2 * math.sqrt(2)), .7071067811865475)
check("72 百万向量GiB", 1000000 * 768 * 4 / 2**30, 2.86102294921875)
check("73 平均倒数排名", (1 + .5 + .25) / 3, .5833333333333334)
check("76 五次摘要节省token", (12000 - 800) * 5, 56000)
check("77 动作净价值甲", .4 * 80 - 12, 20)
check("77 动作净价值乙", .7 * 80 - 60, -4)
check("78 理想并行秒", 20 + 80 / 4, 40)
check("78 含协调秒", 20 + 80 / 4 + 10, 50)
check("79 变化文档", 3000 * .02, 60)
check("79 文本工作量比例", (3000 * 400) / (60 * 400), 50)
check("80 二十步条件通过率", .98**20, .6676079717550942)
check("80 每成功任务成本甲", 100 / 80, 1.25)
check("80 每成功任务成本乙", 180 / 90, 2)

root = Path(__file__).resolve().parent
programs = []
for name in ["structured_tools_demo.py", "offline_rag.py", "agent_state_machine.py"]:
    result = subprocess.run([sys.executable, str(root / name)], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    programs.append({"file": name, "result": payload})
print(json.dumps({"passed": True, "numerical_checks": len(checks), "programs": len(programs),
                  "scope": "representative arithmetic and deterministic offline examples, not trained-model evaluation"}, ensure_ascii=False))
