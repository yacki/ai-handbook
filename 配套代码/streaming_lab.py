"""第68—69章离线实验：SSE碎片、ASGI断开、有限队列、取消与回收。

Python 3.9+，仅用标准库。MockEngine不调用GPU；FakeTransport不使用真实网络。
此例验证应用层生命周期，不代表完成真实推理引擎/代理/浏览器兼容性测试。
"""
import asyncio
import codecs
import json
from itertools import chain
from dataclasses import dataclass


def sse(sequence, event, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {sequence}\nevent: {event}\ndata: {data}\n\n".encode("utf-8")


def parse_our_sse(chunks):
    """仅解析本例的单行JSON数据协议；不是完整的通用EventSource实现。"""
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending, result, previous = "", [], 0
    for chunk in chain(chunks, (None,)):
        pending += decoder.decode(chunk if chunk is not None else b"", final=(chunk is None))
        while "\n\n" in pending:
            frame, pending = pending.split("\n\n", 1)
            if len(frame.encode("utf-8")) > 65536:
                raise ValueError("frame exceeds limit")
            fields = {}
            for line in frame.splitlines():
                if line.startswith(":"):
                    continue
                key, separator, value = line.partition(":")
                if separator:
                    fields[key] = value.lstrip(" ")
            if "data" not in fields:
                continue
            sequence = int(fields["id"])
            if sequence <= previous:
                continue
            if sequence != previous + 1:
                raise ValueError("event gap: resume or snapshot required")
            previous = sequence
            result.append((sequence, fields.get("event", "message"), json.loads(fields["data"])))
        if len(pending.encode("utf-8")) > 65536:
            raise ValueError("pending frame exceeds limit")
    if pending:
        raise ValueError("truncated event frame")
    return result


class SlowConsumer(RuntimeError):
    pass


class MeasuredQueue(asyncio.Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize=maxsize)
        self.high_water = 0

    def put_nowait(self, value):
        super().put_nowait(value)
        self.high_water = max(self.high_water, self.qsize())


class MockEngine:
    def __init__(self, count=40, step_seconds=0.001, put_timeout=0.025):
        self.count, self.step_seconds, self.put_timeout = count, step_seconds, put_timeout
        self.done = asyncio.Event()
        self.error = None
        self.task = None
        self.abort_requested = False
        self.generated = self.high_water = self.live_blocks = self.releases = self.aborts = 0

    def start(self, queue):
        self.task = asyncio.create_task(self._run(queue))

    async def _run(self, queue):
        self.live_blocks = 128  # 模拟资源所有权，不分配真实KV缓存。
        pieces = ["你", "好", "，", "世", "界", "🌍"]
        try:
            for i in range(1, self.count + 1):
                await asyncio.sleep(self.step_seconds)
                self.generated += 1
                event = sse(i, "delta", {"sequence": i, "text": pieces[(i - 1) % len(pieces)]})
                try:
                    await asyncio.wait_for(queue.put(event), self.put_timeout)
                except asyncio.TimeoutError as error:
                    raise SlowConsumer("bounded output queue stayed full") from error
                self.high_water = queue.high_water
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = error
        finally:
            self.live_blocks = 0
            self.releases += 1
            self.done.set()

    async def abort(self):
        if self.task is not None and not self.task.done() and not self.abort_requested:
            self.abort_requested = True
            self.aborts += 1
            self.task.cancel()
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)


class StreamingApp:
    def __init__(self, engine, queue_size=4, send_timeout=0.2):
        self.engine, self.queue_size, self.send_timeout = engine, queue_size, send_timeout
        self.outcome, self.reason, self.terminal_updates = None, None, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            raise ValueError("HTTP only")
        # 先读完请求体，之后只有一个接收协程监听断开。
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                self.outcome, self.reason = "cancelled", "before_admission"
                self.terminal_updates += 1
                return
            if not message.get("more_body", False):
                break
        queue = MeasuredQueue(maxsize=self.queue_size)
        workers = []
        outcome, reason = "failed", "unclassified"
        self.engine.start(queue)

        async def watch_disconnect():
            while (await receive())["type"] != "http.disconnect":
                pass

        async def stream():
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-type", b"text/event-stream; charset=utf-8"),
                (b"cache-control", b"no-cache"),
                (b"x-accel-buffering", b"no"),
            ]})
            while not (self.engine.done.is_set() and queue.empty()):
                if self.engine.error:
                    raise self.engine.error
                try:
                    event = await asyncio.wait_for(queue.get(), 0.02)
                except asyncio.TimeoutError:
                    continue
                await asyncio.wait_for(send({"type": "http.response.body", "body": event, "more_body": True}), self.send_timeout)
            if self.engine.error:
                raise self.engine.error
            final = sse(self.engine.count + 1, "complete", {"status": "completed"})
            await asyncio.wait_for(send({"type": "http.response.body", "body": final, "more_body": True}), self.send_timeout)
            await asyncio.wait_for(send({"type": "http.response.body", "body": b"", "more_body": False}), self.send_timeout)

        try:
            sender = asyncio.create_task(stream())
            watcher = asyncio.create_task(watch_disconnect())
            workers = [sender, watcher]
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            if sender in done:
                await sender
                outcome, reason = "completed", "server_stream_finished"
            else:
                outcome, reason = "cancelled", "http_disconnect"
        except asyncio.CancelledError:
            outcome, reason = "cancelled", "owner_cancelled"
            raise
        except SlowConsumer:
            outcome, reason = "failed", "slow_consumer"
        except asyncio.TimeoutError:
            outcome, reason = "failed", "send_timeout"
        except OSError:
            outcome, reason = "cancelled", "send_failed"
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.engine.abort()
            while not queue.empty():
                queue.get_nowait()
            self.outcome, self.reason = outcome, reason
            self.terminal_updates += 1


@dataclass
class FakeTransport:
    delay: float = 0.0
    disconnect_after: int = 0
    fail_send_after: int = 0

    def __post_init__(self):
        self.disconnected = asyncio.Event()
        self.body_read, self.delta_count, self.frames = False, 0, []

    async def receive(self):
        if not self.body_read:
            self.body_read = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        if self.fail_send_after and self.delta_count >= self.fail_send_after:
            raise OSError("simulated closed transport")
        await asyncio.sleep(self.delay)
        body = message["body"]
        self.frames.append(body)
        if b"event: delta" in body:
            self.delta_count += 1
            if self.disconnect_after and self.delta_count >= self.disconnect_after:
                self.disconnected.set()


async def verify():
    frame = sse(1, "delta", {"text": "你好🌍"}) + sse(2, "complete", {"status": "completed"})
    chunks = [frame[i:i+1] for i in range(len(frame))]
    parsed = parse_our_sse(chunks)
    assert parsed[0][2]["text"] == "你好🌍" and len(parsed) == 2
    cases = [
        ("normal", FakeTransport(), "completed", 0.025, 0.2),
        ("disconnect", FakeTransport(disconnect_after=3), "cancelled", 0.025, 0.2),
        ("send_failure", FakeTransport(fail_send_after=2), "cancelled", 0.025, 0.2),
        ("slow_consumer", FakeTransport(delay=0.06), "failed", 0.025, 0.2),
        ("send_timeout", FakeTransport(delay=0.1), "failed", 0.2, 0.01),
    ]
    report = {"one_byte_utf8_and_sse_chunks": "passed", "cases": []}
    for name, transport, expected, put_timeout, send_timeout in cases:
        engine = MockEngine(put_timeout=put_timeout)
        app = StreamingApp(engine, send_timeout=send_timeout)
        await asyncio.wait_for(app({"type": "http"}, transport.receive, transport.send), 3)
        await engine.abort()  # 重复清理：不再释放一次、不产生新的取消指令。
        assert app.outcome == expected, (name, app.outcome)
        assert engine.live_blocks == 0 and engine.releases == 1
        assert engine.high_water <= 4 and app.terminal_updates == 1
        report["cases"].append({"case": name, "outcome": app.outcome, "reason": app.reason,
            "generated": engine.generated, "delivered_deltas": transport.delta_count,
            "queue_high_water": engine.high_water, "remaining_blocks": engine.live_blocks,
            "resource_releases": engine.releases, "engine_aborts": engine.aborts})
    engine = MockEngine()
    app = StreamingApp(engine)
    async def already_closed():
        return {"type": "http.disconnect"}
    async def unused_send(_message):
        raise AssertionError("no output is allowed before admission")
    await app({"type": "http"}, already_closed, unused_send)
    assert engine.task is None and engine.live_blocks == 0 and app.outcome == "cancelled"
    report["cancel_before_admission"] = "passed: no engine task allocated"
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))
