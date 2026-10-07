"""Task 1 tests: PocketHelperEngine against an in-thread fake helper.

No real process ever starts: engine spawning is faked, HTTP is a local
stub server speaking the helper's POST /tts protocol.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeHelperState:
    def __init__(self):
        self.bodies = []
        self.tts_status = 200
        self.tts_payload = b"\x00\x01\x02\x03" * 480  # 1920 f32 samples
        self.kill_connection_first = False
        self.requests = 0


def make_server(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/tts":
                self.send_response(404)
                self.end_headers()
                return
            state.requests += 1
            if state.kill_connection_first and state.requests == 1:
                self.connection.close()
                return
            length = int(self.headers.get("Content-Length", 0))
            state.bodies.append(json.loads(self.rfile.read(length) or b"{}"))
            self.send_response(state.tts_status)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            if state.tts_status == 200:
                self.wfile.write(state.tts_payload)

    return Handler


class StubProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def poll(self):
        return None if not self.terminated else 0


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from verbix.voice import pocket_helper as mod

    root = tmp_path / "tts"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "verbix-tts").write_bytes(b"fakebinary")
    monkeypatch.setenv("VERBIX_TTS_DIR", str(root))

    state = FakeHelperState()
    spawns = []

    class FakeSpawn:
        def __init__(self, argv, **kwargs):
            port = int(argv[argv.index("--port") + 1])
            spawns.append(argv)
            for _ in range(50):
                try:
                    self.server = ThreadingHTTPServer(("127.0.0.1", port), make_server(state))
                    break
                except OSError:
                    import time

                    time.sleep(0.01)
            else:
                raise AssertionError("fake helper could not bind")
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.proc = StubProcess()
            self.server.proc = self.proc

        def terminate(self):
            self.server.shutdown()
            self.server.server_close()
            self.proc.terminate()

        def kill(self):
            self.terminate()

        def poll(self):
            return self.proc.poll()

    monkeypatch.setattr(mod.subprocess, "Popen", FakeSpawn)
    return state, spawns


def test_synthesize_posts_text_and_yields_pcm_verbatim(rig):
    from verbix.voice.pocket_helper import PocketHelperEngine

    state, spawns = rig

    async def go():
        eng = PocketHelperEngine()
        await eng.load()
        assert eng.is_loaded
        out = [c async for c in eng.synthesize("hello world")]
        await eng.aclose()
        return out

    out = asyncio.run(go())
    assert b"".join(out) == state.tts_payload
    assert state.bodies and state.bodies[0]["text"] == "hello world"
    assert state.bodies[0]["voice"] == "alba.wav"
    assert len(spawns) == 1
    assert spawns[0][0].endswith("verbix-tts")


def test_empty_text_yields_nothing_and_never_spawns(rig):
    from verbix.voice.pocket_helper import PocketHelperEngine

    _, spawns = rig

    async def go():
        eng = PocketHelperEngine()
        out = [c async for c in eng.synthesize("   \n ")]
        await eng.aclose()
        return out, eng.is_loaded

    out, loaded = asyncio.run(go())
    assert out == []
    assert not loaded
    assert spawns == []


def test_non200_raises_naming_chunk_and_status(rig):
    from verbix.voice.pocket_helper import PocketHelperEngine

    state, _ = rig
    state.tts_status = 500

    async def go():
        eng = PocketHelperEngine()
        await eng.load()
        try:
            async for _ in eng.synthesize("boom"):
                pass
            return None
        except RuntimeError as e:
            return str(e)
        finally:
            await eng.aclose()

    msg = asyncio.run(go())
    assert msg is not None and "500" in msg


def test_dead_helper_restarts_once_then_errors(rig):
    from verbix.voice.pocket_helper import PocketHelperEngine

    state, spawns = rig
    state.kill_connection_first = True

    async def go():
        eng = PocketHelperEngine()
        await eng.load()
        out = [c async for c in eng.synthesize("again")]
        await eng.aclose()
        return out

    out = asyncio.run(go())
    assert b"".join(out) == state.tts_payload
    assert len(spawns) == 2


def test_double_load_starts_one_helper_and_aclose_kills_it(rig):
    from verbix.voice.pocket_helper import PocketHelperEngine

    _, spawns = rig
    holder = {}

    async def go():
        eng = PocketHelperEngine()
        await eng.load()
        await eng.load()
        holder["eng"] = eng
        return len(spawns)

    assert asyncio.run(go()) == 1
    eng = holder["eng"]

    async def bye():
        await eng.aclose()
        await eng.aclose()

    asyncio.run(bye())
    assert not eng.is_loaded
