"""第81章离线harness实验：事件去重/订阅直通/持久工具回执。

Python 3.9+标准库；不安装DSH，不连接GitHub/飞书，不发送真实消息。
验证指定故障边界，不是完整分布式系统或智能体能力评测。
"""
import json
import sqlite3
import tempfile
from pathlib import Path


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Inbox:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS triggers(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS subscriptions(thread TEXT PRIMARY KEY, worker TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS deliveries(worker TEXT, event TEXT, PRIMARY KEY(worker,event));
        """)
        self.router_calls = self.direct_deliveries = 0

    def receive(self, event):
        encoded = compact(event)
        with self.db:
            old = self.db.execute("SELECT body FROM triggers WHERE id=?", (event["id"],)).fetchone()
            if old:
                if old[0] != encoded:
                    raise ValueError("same event ID with different payload")
                return "duplicate"
            self.db.execute("INSERT INTO triggers VALUES(?,?)", (event["id"], encoded))
            # 此处的白名单是本实验策略，不是从文字内容推断权限。
            if event["actor"] != "alice" or event["source"] not in {"im", "github"}:
                return "filtered"
            row = self.db.execute("SELECT worker FROM subscriptions WHERE thread=?", (event["thread"],)).fetchone()
            if row:
                worker = row[0]
                self.direct_deliveries += 1
            else:
                if not event["requested"]:
                    return "recorded_without_action"
                # 确定性路由替身；此处不运行LLM语义判断。
                self.router_calls += 1
                worker = "worker:" + event["thread"]
                self.db.execute("INSERT INTO subscriptions VALUES(?,?)", (event["thread"], worker))
            self.db.execute("INSERT INTO deliveries VALUES(?,?)", (worker, event["id"]))
            return worker


class ToolService:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS inventory(item TEXT PRIMARY KEY, remaining INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY, arguments TEXT NOT NULL, result TEXT NOT NULL);
        INSERT OR IGNORE INTO inventory VALUES('teaching-item',10);
        """)

    def reserve(self, operation_key, quantity):
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise ValueError("positive integer quantity required")
        args = compact({"item": "teaching-item", "quantity": quantity})
        with self.db:
            # 显式取得写锁：回执检查、库存更新与回执写入处在同一事务。
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT arguments,result FROM receipts WHERE key=?", (operation_key,)).fetchone()
            if row:
                if row[0] != args:
                    raise ValueError("idempotency key reused with different arguments")
                return json.loads(row[1])
            remaining = self.db.execute("SELECT remaining FROM inventory WHERE item='teaching-item'").fetchone()[0]
            if quantity > remaining:
                raise ValueError("insufficient stock")
            result = {"remaining": remaining - quantity, "quantity": quantity}
            self.db.execute("UPDATE inventory SET remaining=? WHERE item='teaching-item'", (result["remaining"],))
            self.db.execute("INSERT INTO receipts VALUES(?,?,?)", (operation_key, args, compact(result)))
            return result


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")

    def append(self, event_id, kind, payload):
        with self.db:
            old = self.db.execute("SELECT kind,payload FROM events WHERE id=?", (event_id,)).fetchone()
            value = compact(payload)
            if old and old != (kind, value):
                raise ValueError("conflicting journal event")
            self.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?)", (event_id, kind, value))

    def pending_operations(self):
        planned, completed = {}, set()
        for kind, payload in self.db.execute("SELECT kind,payload FROM events ORDER BY rowid"):
            value = json.loads(payload)
            key = value["operation_key"]
            if kind == "tool_planned":
                planned[key] = value
            elif kind == "tool_completed":
                completed.add(key)
        return [value for key, value in planned.items() if key not in completed]


def verify():
    # 文件限定在临时目录，退出即清理；所有动作均为虚构业务数据。
    with tempfile.TemporaryDirectory(prefix="ai-book-harness-") as directory:
        base = Path(directory)
        inbox = Inbox(base / "inbox.sqlite")
        first = {"id": "e1", "thread": "pr:7", "actor": "alice", "source": "github", "requested": True}
        follow = {"id": "e2", "thread": "pr:7", "actor": "alice", "source": "im", "requested": False}
        rejected = {"id": "e3", "thread": "pr:8", "actor": "mallory", "source": "github", "requested": True}
        assert inbox.receive(first) == "worker:pr:7"
        assert inbox.receive(follow) == "worker:pr:7"
        assert inbox.receive(follow) == "duplicate"
        assert inbox.receive(rejected) == "filtered"
        assert inbox.router_calls == 1 and inbox.direct_deliveries == 1

        journal = Journal(base / "journal.sqlite")
        tool = ToolService(base / "tool.sqlite")
        key = "tenant-teaching/run-1/call-1"
        journal.append("planned-1", "tool_planned", {"operation_key": key, "quantity": 3})
        tool.reserve(key, 3)
        # 模拟工具已提交，但harness尚未保存返回结果就退出。
        tool.db.close()
        journal.db.close()
        recovered = Journal(base / "journal.sqlite")
        new_tool_connection = ToolService(base / "tool.sqlite")
        pending = recovered.pending_operations()
        assert len(pending) == 1
        command = pending[0]
        replay_result = new_tool_connection.reserve(command["operation_key"], command["quantity"])
        assert replay_result == {"remaining": 7, "quantity": 3}
        recovered.append("result-1", "tool_completed", {"operation_key": command["operation_key"], "result": replay_result})
        assert recovered.pending_operations() == []
        receipts = new_tool_connection.db.execute("SELECT count(*) FROM receipts").fetchone()[0]
        remaining = new_tool_connection.db.execute("SELECT remaining FROM inventory").fetchone()[0]
        assert receipts == 1 and remaining == 7
        try:
            new_tool_connection.reserve(key, 4)
        except ValueError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        assert mismatch_rejected
        report = {"router_decisions": 1, "direct_deliveries": 1, "duplicate_event": "ignored",
            "unauthorized_actor": "filtered", "stock_after_commit_and_recovery": remaining,
            "durable_tool_receipts": receipts, "changed_arguments_with_same_key": "rejected",
            "scope": "离线确定性调度与指定崩溃窗口；不运行真实模型、DSH或外部工具"}
        inbox.db.close()
        recovered.db.close()
        new_tool_connection.db.close()
        return report


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
