"""Unit tests for the agent harness.

Run with::

    python -m unittest agent_test -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import agent as A


def run(coro):
    return asyncio.run(coro)


class ConfigTest(unittest.TestCase):
    def test_merge_keeps_known_fields_and_collects_extras(self):
        config = A.AgentConfig().merge(model="m", genre="noir", ignored=None)
        self.assertEqual(config.model, "m")
        self.assertEqual(config.extras["genre"], "noir")
        self.assertNotIn("ignored", config.extras)

    def test_extras_are_readable_as_attributes(self):
        config = A.AgentConfig(extras={"genre": "noir"})
        self.assertEqual(config.genre, "noir")
        with self.assertRaises(AttributeError):
            _ = config.missing

    def test_env_layer_coerces_types(self):
        env = {"AGENT_PORT": "9100", "AGENT_QUIET": "true", "AGENT_MODEL": "env-model"}
        config = A.AgentConfig().with_env(env)
        self.assertEqual(config.port, 9100)
        self.assertTrue(config.quiet)
        self.assertEqual(config.model, "env-model")

    def test_env_layer_coerces_paths(self):
        config = A.AgentConfig().with_env({"AGENT_THEME": "/tmp/theme.json"})
        self.assertEqual(config.theme, Path("/tmp/theme.json"))

    def test_to_dict_redacts_secrets(self):
        config = A.AgentConfig(api_key="sk-abcdefghijklmnop")
        self.assertEqual(config.to_dict()["api_key"], "sk-abcde...mnop")
        self.assertEqual(config.to_dict(redact=False)["api_key"], "sk-abcdefghijklmnop")

    def test_to_dict_redacts_role_secrets(self):
        config = A.AgentConfig(vision_api_key="sk-abcdefghijklmnop")
        self.assertEqual(config.to_dict()["vision_api_key"], "sk-abcde...mnop")

    def test_mask_secret(self):
        self.assertEqual(A.mask_secret("short"), "*****")
        self.assertEqual(A.mask_secret(""), "")


class ModelSpecTest(unittest.TestCase):
    def test_leader_spec_uses_the_primary_settings(self):
        config = A.AgentConfig(model="m", api_url="http://url", api_key="k")
        spec = config.model_spec()
        self.assertEqual((spec.role, spec.name, spec.api_url, spec.api_key),
                         ("leader", "m", "http://url", "k"))

    def test_vision_is_unset_by_default(self):
        self.assertIsNone(A.AgentConfig().model_spec(A.VISION_ROLE))

    def test_vision_inherits_the_leader_endpoint(self):
        config = A.AgentConfig(api_url="http://url", api_key="k", vision_model="v")
        spec = config.model_spec(A.VISION_ROLE)
        self.assertEqual((spec.name, spec.api_url, spec.api_key), ("v", "http://url", "k"))

    def test_vision_may_use_its_own_endpoint(self):
        config = A.AgentConfig(
            vision_model="v", vision_api_url="http://other", vision_api_key="k2"
        )
        spec = config.model_spec(A.VISION_ROLE)
        self.assertEqual((spec.api_url, spec.api_key), ("http://other", "k2"))

    def test_vision_model_appears_in_the_banner(self):
        keys = [k for k, _ in A.AgentConfig(vision_model="v").banner_items()]
        self.assertIn("Vision Model", keys)
        self.assertNotIn("Vision Model", [k for k, _ in A.AgentConfig().banner_items()])


class ModelPoolTest(unittest.TestCase):
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.chat = self

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            message = type("M", (), {"content": " a cat "})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    class Pool(A.ModelPool):
        def __init__(self):
            super().__init__()
            self.built = 0

        def client(self, spec):
            client = self._clients.get(spec.endpoint)
            if client is None:
                self.built += 1
                client = ModelPoolTest.FakeClient(base_url=spec.api_url)
                self._clients[spec.endpoint] = client
            return client

    def setUp(self):
        self.pool = self.Pool()

    def test_one_client_per_endpoint(self):
        leader = A.ModelSpec("leader", "m", "http://url", "k")
        vision = A.ModelSpec("vision", "v", "http://url", "k")
        other = A.ModelSpec("vision", "v", "http://other", "k")
        self.assertIs(self.pool.client(leader), self.pool.client(vision))
        self.assertIsNot(self.pool.client(leader), self.pool.client(other))
        self.assertEqual(self.pool.built, 2)

    def test_describe_image_sends_the_data_url_and_returns_text(self):
        spec = A.ModelSpec("vision", "v", "http://url", "k")
        answer = run(self.pool.describe_image(spec, "data:image/png;base64,AA", "what?"))
        self.assertEqual(answer, "a cat")
        payload = self.pool.client(spec).calls[0]
        self.assertEqual(payload["model"], "v")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["text"], "what?")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,AA")

    def test_clear_drops_cached_clients(self):
        spec = A.ModelSpec("leader", "m", "http://url", "k")
        first = self.pool.client(spec)
        self.pool.clear()
        self.assertIsNot(first, self.pool.client(spec))


class InputResolutionTest(unittest.TestCase):
    def test_raw_text_is_returned_as_is(self):
        self.assertEqual(A.resolve_input("just text"), "just text")
        self.assertEqual(A.resolve_input(""), "")

    def test_file_path_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text("from file", encoding="utf-8")
            self.assertEqual(A.resolve_input(str(path)), "from file")

    def test_agent_assigns_resolved_text_to_config_input(self):
        agent = A.Agent(console=False, store=A.MemoryStore())
        self.addCleanup(agent.close)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text("story seed", encoding="utf-8")
            config = agent.resolve_config(input=str(path))
        self.assertEqual(config.input, "story seed")
        self.assertTrue(config.input_source.endswith("prompt.md"))


class EventBusTest(unittest.TestCase):
    def test_sync_and_async_subscribers_receive_events(self):
        bus = A.EventBus()
        seen: list[str] = []
        bus.on("a", lambda e: seen.append(f"sync:{e.data['v']}"))

        async def handler(event):
            seen.append(f"async:{event.data['v']}")

        bus.on("a", handler)
        bus.on(A.EventType.ALL, lambda e: seen.append(f"all:{e.type}"))
        run(bus.publish("a", v=1))
        self.assertEqual(seen, ["sync:1", "async:1", "all:a"])

    def test_unsubscribe_and_failing_subscriber(self):
        bus = A.EventBus()
        seen: list[str] = []
        off = bus.on("a", lambda e: seen.append("first"))
        bus.on("a", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        bus.on("a", lambda e: seen.append("last"))
        off()
        run(bus.publish("a"))
        self.assertEqual(seen, ["last"])  # failure never breaks the chain


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.prompts = A.PromptRenderer()

    def test_renders_config_input(self):
        config = A.AgentConfig(input="the seed")
        rendered = self.prompts.render("Use: {{ config.input }}", config)
        self.assertEqual(rendered, "Use: the seed")

    def test_renders_extras_and_context(self):
        config = A.AgentConfig(extras={"genre": "noir"})
        rendered = self.prompts.render("{{ config.genre }}/{{ tone }}", config, tone="dry")
        self.assertEqual(rendered, "noir/dry")

    def test_strict_undefined_raises(self):
        from jinja2 import UndefinedError

        with self.assertRaises(UndefinedError):
            self.prompts.render("{{ nope }}", A.AgentConfig())

    def test_render_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.md"
            path.write_text("Hi {{ config.input }}", encoding="utf-8")
            self.assertEqual(
                self.prompts.render_file(path, A.AgentConfig(input="there")), "Hi there"
            )


class StorageTest(unittest.TestCase):
    def test_sqlite_store_roundtrip(self):
        store = A.SqliteStore()
        self.addCleanup(store.close)
        store.set("ns", "k", {"a": 1})
        self.assertEqual(store.get("ns", "k"), {"a": 1})
        store.set("ns", "k", {"a": 2})  # upsert
        self.assertEqual(store.list("ns"), [("k", {"a": 2})])
        store.delete("ns", "k")
        self.assertIsNone(store.get("ns", "k"))
        self.assertEqual(store.list("ns"), [])

    def test_sqlite_store_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "agent.db"
            store = A.SqliteStore(path)
            store.set("ns", "k", "v")
            store.close()
            reopened = A.SqliteStore(path)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.get("ns", "k"), "v")

    def test_stores_satisfy_the_protocol(self):
        self.assertIsInstance(A.MemoryStore(), A.Store)
        self.assertIsInstance(A.LruCache(2), A.Cache)

    def test_lru_cache_evicts_least_recently_used(self):
        cache = A.LruCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)  # refresh 'a'
        cache.set("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)
        cache.delete("c")
        self.assertIsNone(cache.get("c"))
        cache.clear()
        self.assertEqual(len(cache), 0)


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.store = A.MemoryStore()
        self.sessions = A.SessionManager(ttl=60, store=self.store, cache=A.LruCache(4))
        self.addCleanup(self.sessions.close_all)

    def test_create_isolates_a_workspace(self):
        first = self.sessions.create()
        second = self.sessions.create()
        self.assertTrue(first.workspace.is_dir())
        self.assertNotEqual(first.workspace, second.workspace)
        self.assertEqual(self.store.get("sessions", first.id)["id"], first.id)

    def test_close_leaves_no_trace(self):
        session = self.sessions.create()
        (session.workspace / "note.txt").write_text("x", encoding="utf-8")
        self.assertTrue(self.sessions.close(session.id))
        self.assertFalse(session.workspace.exists())
        self.assertIsNone(self.store.get("sessions", session.id))
        self.assertFalse(self.sessions.close(session.id))

    def test_expired_sessions_are_purged(self):
        session = self.sessions.create()
        session.expires_at = 0
        self.assertTrue(session.expired)
        self.assertEqual(self.sessions.purge_expired(), [session.id])
        self.assertIsNone(self.sessions.get(session.id))
        self.assertFalse(session.workspace.exists())

    def test_ensure_reuses_a_live_session(self):
        session = self.sessions.create()
        self.assertIs(self.sessions.ensure(session.id).id, session.id)
        self.assertNotEqual(self.sessions.ensure(None).id, session.id)

    def test_keep_workspace_preserves_the_directory(self):
        manager = A.SessionManager(ttl=60, keep_workspace=True)
        session = manager.create()
        self.addCleanup(lambda: __import__("shutil").rmtree(session.workspace, True))
        manager.close(session.id)
        self.assertTrue(session.workspace.exists())

    def test_access_renews_the_lease(self):
        session = self.sessions.create()
        session.expires_at = A.time.time() + 1
        renewed = self.sessions.get(session.id)
        self.assertGreater(renewed.expires_at, A.time.time() + 30)
        self.assertGreater(self.store.get("sessions", session.id)["expires_at"], 0)


class DurableSessionTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, True))
        self.store = A.MemoryStore()

    def manager(self, **overrides):
        options = dict(
            root=self.root, ttl=60, store=self.store, prefix="agent", durable=True
        )
        options.update(overrides)
        manager = A.SessionManager(**options)
        self.addCleanup(lambda: manager.close_all(destroy=True))
        return manager

    def test_a_durable_shutdown_keeps_the_workspace_and_the_record(self):
        first = self.manager()
        session = first.create()
        first.close_all()
        self.assertTrue(session.workspace.is_dir())
        self.assertIsNotNone(self.store.get("sessions", session.id))

    def test_live_sessions_are_rehydrated_on_startup(self):
        first = self.manager()
        session = first.create(who="me")
        (session.workspace / "note.txt").write_text("x", encoding="utf-8")
        first.close_all()
        second = self.manager()
        restored = second.get(session.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.workspace, session.workspace)
        self.assertEqual(restored.meta, {"who": "me"})
        self.assertEqual([s.id for s in second.list()], [session.id])

    def test_expired_and_vanished_records_are_reconciled_away(self):
        first = self.manager()
        stale = first.create()
        stale.expires_at = 0
        first.update(stale)
        vanished = first.create()
        __import__("shutil").rmtree(vanished.workspace)
        first.close_all()
        second = self.manager()
        self.assertEqual(second.list(), [])
        self.assertIsNone(self.store.get("sessions", stale.id))
        self.assertIsNone(self.store.get("sessions", vanished.id))
        self.assertFalse(stale.workspace.exists())

    def test_unreadable_records_are_dropped(self):
        self.store.set("sessions", "broken", {"nope": True})
        self.assertEqual(self.manager().list(), [])
        self.assertIsNone(self.store.get("sessions", "broken"))

    def test_directories_without_an_owner_are_reaped(self):
        manager = self.manager()
        session = manager.create()
        orphan = self.root / "agent-orphan-1"
        orphan.mkdir()
        foreign = self.root / "other-tool-1"
        foreign.mkdir()
        self.assertEqual(manager.orphans(), [orphan])
        self.assertEqual(manager.reap_orphans(), [orphan])
        self.assertFalse(orphan.exists())
        self.assertTrue(foreign.is_dir())
        self.assertTrue(session.workspace.is_dir())

    def test_kept_workspaces_are_never_reaped(self):
        manager = self.manager(keep_workspace=True)
        orphan = self.root / "agent-orphan-1"
        orphan.mkdir()
        self.assertEqual(manager.reap_orphans(), [])
        self.assertTrue(orphan.is_dir())

    def test_sweep_expires_and_reaps(self):
        manager = self.manager()
        session = manager.create()
        session.expires_at = 0
        orphan = self.root / "agent-orphan-1"
        orphan.mkdir()
        expired, reaped = manager.sweep()
        self.assertEqual(expired, [session.id])
        self.assertEqual(reaped, [orphan])
        self.assertFalse(session.workspace.exists())

    def test_the_sweeper_expires_on_a_timer(self):
        async def scenario():
            manager = self.manager(sweep_interval=0.01)
            session = manager.create()
            session.expires_at = 0
            manager.start_sweeper()
            self.assertIs(manager.start_sweeper(), manager._sweeper)
            for _ in range(200):
                await asyncio.sleep(0.01)
                if not manager.list():
                    break
            await manager.stop_sweeper()
            return session

        session = run(scenario())
        self.assertFalse(session.workspace.exists())

    def test_the_sweeper_is_optional(self):
        async def scenario():
            manager = self.manager(sweep_interval=0)
            self.assertIsNone(manager.start_sweeper())
            await manager.stop_sweeper()

        run(scenario())


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.store = A.MemoryStore()
        self.memory = A.ConversationMemory(self.store, max_turns=4, max_chars=0)

    def exchange(self, asked, answered):
        return run(
            self.memory.remember("s1", [A.Turn("user", asked), A.Turn("assistant", answered)])
        )

    def test_remembered_turns_become_chat_messages(self):
        self.exchange("hello", "hi")
        self.assertEqual(
            self.memory.history("s1"),
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )

    def test_empty_turns_are_not_remembered(self):
        self.exchange("hello", "   ")
        self.assertEqual(len(self.memory.transcript("s1")), 1)

    def test_transcripts_survive_a_new_memory_over_the_same_store(self):
        self.exchange("hello", "hi")
        reloaded = A.ConversationMemory(self.store)
        self.assertEqual(reloaded.transcript("s1").messages(), self.memory.history("s1"))

    def test_replay_carries_the_block_kind_of_each_role(self):
        self.exchange("hello", "hi")
        self.assertEqual([t["kind"] for t in self.memory.replay("s1")], ["prompt", "output"])

    def test_oldest_turns_fall_out_of_the_window(self):
        self.exchange("one", "1")
        self.exchange("two", "2")
        self.exchange("three", "3")
        self.assertEqual(
            [t.text for t in self.memory.transcript("s1").turns], ["two", "2", "three", "3"]
        )

    def test_a_character_budget_also_trims(self):
        memory = A.ConversationMemory(self.store, max_turns=0, max_chars=6)
        run(memory.remember("s2", [A.Turn("user", "aaaa"), A.Turn("assistant", "bbbb")]))
        self.assertEqual([t.text for t in memory.transcript("s2").turns], ["bbbb"])

    def test_the_newest_turn_always_survives(self):
        memory = A.ConversationMemory(self.store, max_turns=0, max_chars=1)
        run(memory.remember("s3", [A.Turn("user", "a long question")]))
        self.assertEqual(len(memory.transcript("s3")), 1)

    def test_a_summarizer_replaces_what_falls_out(self):
        async def summarize(turns):
            return "gist: " + ",".join(t.text for t in turns)

        memory = A.ConversationMemory(self.store, max_turns=2, summarizer=summarize)
        run(memory.remember("s4", [A.Turn("user", "one"), A.Turn("assistant", "1")]))
        run(memory.remember("s4", [A.Turn("user", "two"), A.Turn("assistant", "2")]))
        turns = memory.transcript("s4").turns
        self.assertEqual(turns[0].role, "system")
        self.assertEqual(turns[0].text, "gist: one,1")
        self.assertEqual([t.text for t in turns[1:]], ["two", "2"])

    def test_a_failing_summarizer_only_costs_the_summary(self):
        async def summarize(_turns):
            raise RuntimeError("boom")

        memory = A.ConversationMemory(self.store, max_turns=2, summarizer=summarize)
        run(memory.remember("s5", [A.Turn("user", "one"), A.Turn("assistant", "1")]))
        run(memory.remember("s5", [A.Turn("user", "two"), A.Turn("assistant", "2")]))
        self.assertEqual([t.text for t in memory.transcript("s5").turns], ["two", "2"])

    def test_forget_erases_the_transcript(self):
        self.exchange("hello", "hi")
        self.memory.forget("s1")
        self.assertEqual(self.memory.history("s1"), [])
        self.assertIsNone(self.store.get(A.ConversationMemory.NAMESPACE, "s1"))

    def test_disabled_memory_remembers_nothing(self):
        memory = A.ConversationMemory(self.store, enabled=False)
        run(memory.remember("s6", [A.Turn("user", "hello")]))
        self.assertEqual(memory.history("s6"), [])
        self.assertEqual(memory.replay("s6"), [])
        self.assertIsNone(self.store.get(A.ConversationMemory.NAMESPACE, "s6"))


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = A.Workspace(self.tmp.name, shell_timeout=10)

    def test_paths_are_scoped_to_the_root(self):
        self.assertEqual(self.ws.resolve("a/b.txt"), self.ws.root / "a" / "b.txt")
        for escape in ("../outside.txt", "/etc/passwd", "a/../../x", ""):
            with self.assertRaises(A.WorkspaceError):
                self.ws.resolve(escape)

    def test_symlinks_cannot_escape_the_root(self):
        (Path(self.ws.root) / "link").symlink_to(Path(self.tmp.name).parent)
        with self.assertRaises(A.WorkspaceError):
            self.ws.resolve("link/outside.txt")

    def test_quotes_are_stripped(self):
        self.assertEqual(self.ws.resolve("'a.txt'"), self.ws.root / "a.txt")

    def test_read_write_and_list(self):
        self.ws.write_text("dir/file.txt", "hello")
        self.assertEqual(self.ws.read_text("dir/file.txt"), "hello")
        self.assertEqual(self.ws.list_dir("."), ["dir/"])
        with self.assertRaises(A.WorkspaceError):
            self.ws.read_text("missing.txt")
        with self.assertRaises(A.WorkspaceError):
            self.ws.list_dir("missing")

    def test_read_data_url(self):
        (Path(self.ws.root) / "pixel.png").write_bytes(b"\x89PNG\r\n")
        url = self.ws.read_data_url("pixel.png")
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), b"\x89PNG\r\n")

    def test_empty_file_is_rejected_for_data_url(self):
        (Path(self.ws.root) / "empty.png").write_bytes(b"")
        with self.assertRaises(A.WorkspaceError):
            self.ws.read_data_url("empty.png")

    def test_shell_runs_in_the_workspace(self):
        output = run(self.ws.run_shell("pwd"))
        self.assertIn(str(self.ws.root), output)

    def test_shell_reports_failures(self):
        self.assertIn("[exit code 3]", run(self.ws.run_shell("exit 3")))

    def test_shell_times_out(self):
        ws = A.Workspace(self.tmp.name, shell_timeout=1)
        self.assertIn("timed out", run(ws.run_shell("sleep 5")))

    def test_guess_mime(self):
        self.assertEqual(A.guess_mime("a.png"), "image/png")
        self.assertEqual(A.guess_mime("a.unknown"), "application/octet-stream")


class ToolRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = A.Workspace(self.tmp.name)
        (Path(self.ws.root) / "a.png").write_bytes(b"\x89PNG")

    @staticmethod
    def invoke(tool, payload):
        from agents.tool_context import ToolContext

        context = ToolContext(
            context=None, tool_name=tool.name, tool_call_id="1", tool_arguments=payload
        )
        return run(tool.on_invoke_tool(context, payload))

    def tools(self, vision=None):
        built = A.ToolRegistry(shell=False).build(self.ws, vision=vision)
        return {tool.name: tool for tool in built}

    def test_leader_sees_images_when_no_vision_model_is_configured(self):
        tools = self.tools()
        self.assertIn("view_image", tools)
        self.assertNotIn("describe_image", tools)
        self.assertTrue(
            self.invoke(tools["view_image"], '{"path": "a.png"}').startswith("data:image/png")
        )

    def test_vision_model_replaces_the_raw_image_tool(self):
        seen: list[tuple[str, str]] = []

        async def vision(data_url, question):
            seen.append((data_url, question))
            return "a cat"

        tools = self.tools(vision)
        self.assertIn("describe_image", tools)
        self.assertNotIn("view_image", tools)
        answer = self.invoke(tools["describe_image"], '{"path": "a.png", "question": "q"}')
        self.assertEqual(answer, "a cat")
        self.assertTrue(seen[0][0].startswith("data:image/png"))
        self.assertEqual(seen[0][1], "q")

    def test_vision_errors_are_reported_to_the_leader(self):
        async def vision(data_url, question):
            raise RuntimeError("upstream is down")

        tools = self.tools(vision)
        answer = self.invoke(tools["describe_image"], '{"path": "a.png"}')
        self.assertEqual(answer, "Error: the vision model failed: upstream is down")

    def test_registered_tools_are_appended(self):
        registry = A.ToolRegistry(defaults=False, shell=False)
        registry.register("mine", lambda ws: f"tool:{ws.root.name}")
        self.assertEqual(registry.build(self.ws), [f"tool:{self.ws.root.name}"])


class RepoSpecTest(unittest.TestCase):
    def test_https_ssh_and_shorthand_urls_resolve_to_the_same_repository(self):
        for url in (
            "https://github.com/owner/name.git",
            "git@github.com:owner/name.git",
            "owner/name",
        ):
            spec = A.RepoSpec(url=url)
            self.assertEqual(spec.slug, "owner/name", url)
            self.assertEqual(spec.name, "name", url)
            self.assertEqual(spec.api_url, "https://api.github.com", url)

    def test_enterprise_host_uses_its_own_api(self):
        spec = A.RepoSpec(url="https://git.acme.io/owner/name")
        self.assertEqual(spec.api_url, "https://git.acme.io/api/v3")

    def test_unknown_urls_are_rejected(self):
        with self.assertRaises(A.RepoError):
            _ = A.RepoSpec(url="not a repository").slug
        with self.assertRaises(A.RepoError):
            _ = A.RepoSpec().slug

    def test_filesystem_repositories_have_no_pull_request_api(self):
        spec = A.RepoSpec(url="/tmp/origin.git")
        self.assertTrue(spec.local)
        self.assertEqual(spec.name, "origin")
        self.assertEqual(spec.clone_url, "/tmp/origin.git")
        with self.assertRaises(A.RepoError):
            _ = spec.api_url

    def test_token_is_embedded_only_in_the_authenticated_url_and_masked(self):
        spec = A.RepoSpec(url="https://github.com/owner/name.git", token="ghp_secret")
        self.assertNotIn("ghp_secret", spec.clone_url)
        self.assertIn("x-access-token:ghp_secret@", spec.authenticated_url())
        self.assertEqual(spec.mask("fatal: ghp_secret"), "fatal: ***")

    def test_config_layers_into_a_spec(self):
        config = A.AgentConfig().with_env(
            {"AGENT_REPO_URL": "owner/name", "AGENT_REPO_BRANCH": "trunk"}
        )
        spec = A.RepoSpec.from_config(config)
        self.assertEqual((spec.url, spec.branch), ("owner/name", "trunk"))
        self.assertEqual(config.to_dict()["repo_token"], "")


def git(*args, cwd):
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "HOME": str(cwd),
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


class RepoManagerTest(unittest.TestCase):
    """Clone, commit and push against a bare repository on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.root / "origin.git"
        seed = self.root / "seed"
        seed.mkdir()
        git("init", "--initial-branch", "main", cwd=seed)
        (seed / "README.md").write_text("seed", encoding="utf-8")
        git("add", "--all", cwd=seed)
        git("commit", "--message", "seed", cwd=seed)
        git("init", "--bare", "--initial-branch", "main", str(self.origin), cwd=self.root)
        git("remote", "add", "origin", str(self.origin), cwd=seed)
        git("push", "origin", "main", cwd=seed)
        self.manager = A.RepoManager(A.RepoSpec(url=str(self.origin)), timeout=60)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def clone(self, session_id="abc123"):
        return run(self.manager.clone(self.workspace, session_id=session_id))

    def test_clone_checks_out_a_session_branch(self):
        checkout = self.clone()
        self.assertEqual(checkout.path, self.workspace / "origin")
        self.assertEqual(checkout.branch, "agent/abc123")
        self.assertEqual(checkout.base, "main")
        self.assertTrue((checkout.path / "README.md").is_file())
        self.assertEqual(
            run(self.manager.git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout.path)),
            "agent/abc123",
        )

    def test_clone_refuses_a_non_empty_target(self):
        self.clone()
        with self.assertRaises(A.RepoError):
            self.clone()

    def test_commit_and_push_reach_the_remote(self):
        checkout = self.clone()
        (checkout.path / "note.txt").write_text("work", encoding="utf-8")
        self.assertIn("add note", run(checkout.commit("add note")))
        run(checkout.push())
        branches = run(
            self.manager.git("ls-remote", "--heads", str(self.origin), cwd=self.workspace)
        )
        self.assertIn("agent/abc123", branches)

    def test_commit_without_changes_is_refused(self):
        checkout = self.clone()
        with self.assertRaises(A.RepoError):
            run(checkout.commit("nothing"))
        with self.assertRaises(A.RepoError):
            run(checkout.commit("   "))

    def test_status_reports_the_working_tree(self):
        checkout = self.clone()
        (checkout.path / "note.txt").write_text("work", encoding="utf-8")
        self.assertIn("note.txt", run(checkout.status()))

    def test_failing_git_commands_raise(self):
        with self.assertRaises(A.RepoError):
            run(self.manager.git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.workspace))

    def test_pull_request_posts_to_the_forge(self):
        posted: list[tuple[str, dict]] = []

        class Forge(A.RepoManager):
            def _post(self, url, payload):
                posted.append((url, payload))
                return {"number": 7, "html_url": "https://example/pull/7", "state": "open"}

        manager = Forge(A.RepoSpec(url="owner/name", token="t"), timeout=5)
        pull = run(
            manager.open_pull_request(title="Work", body="b", head="agent/x", base="main")
        )
        self.assertEqual(pull["url"], "https://example/pull/7")
        self.assertEqual(posted[0][0], "https://api.github.com/repos/owner/name/pulls")
        self.assertEqual(posted[0][1]["head"], "agent/x")

    def test_pull_request_requires_a_token_a_title_and_a_base(self):
        manager = A.RepoManager(A.RepoSpec(url="owner/name"))
        for kwargs in (
            {"title": "", "head": "h", "base": "main"},
            {"title": "t", "head": "h", "base": "main"},
        ):
            with self.assertRaises(A.RepoError):
                run(manager.open_pull_request(**kwargs))
        manager.spec = A.RepoSpec(url="owner/name", token="t")
        with self.assertRaises(A.RepoError):
            run(manager.open_pull_request(title="t", head="h", base=""))


class AgentRepoTest(unittest.TestCase):
    """The session lifecycle around a repository backed workspace."""

    class LocalSpec(A.RepoSpec):
        """A filesystem repository that pretends to have a forge API."""

        @property
        def api_url(self) -> str:
            return "https://api.example"

    class Forge(A.RepoManager):
        def _post(self, url, payload):
            return {"number": 3, "html_url": "https://example/pull/3", "state": "open"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.root / "origin.git"
        seed = self.root / "seed"
        seed.mkdir()
        git("init", "--initial-branch", "main", cwd=seed)
        (seed / "README.md").write_text("seed", encoding="utf-8")
        git("add", "--all", cwd=seed)
        git("commit", "--message", "seed", cwd=seed)
        git("init", "--bare", "--initial-branch", "main", str(self.origin), cwd=self.root)
        git("remote", "add", "origin", str(self.origin), cwd=seed)
        git("push", "origin", "main", cwd=seed)
        self.agent = A.Agent(
            console=False,
            store=A.MemoryStore(),
            repos=self.Forge(timeout=60),
            repo_url=str(self.origin),
            repo_token="t",
        )
        self.agent.repos.spec = self.LocalSpec(url=str(self.origin), token="t")
        self.addCleanup(self.agent.close)

    def test_new_session_clones_the_default_repository(self):
        result = run(self.agent.commands.invoke("/new"))
        session = self.agent.sessions.get(result["session"]["id"])
        checkout = self.agent.checkout(session)
        self.assertIsNotNone(checkout)
        self.assertEqual(checkout.branch, f"agent/{session.id}")
        self.assertTrue((checkout.path / "README.md").is_file())
        self.assertEqual(session.meta["repo"]["base"], "main")

    def test_prepare_session_is_idempotent(self):
        session = self.agent.sessions.create()
        first = run(self.agent.prepare_session(session))
        second = run(self.agent.prepare_session(session))
        self.assertEqual(first.path, second.path)

    def test_a_clone_failure_leaves_the_session_usable(self):
        self.agent.repos.spec = A.RepoSpec(url="/nowhere/missing.git")
        session = self.agent.sessions.create()
        self.assertIsNone(run(self.agent.prepare_session(session)))
        self.assertIsNone(self.agent.checkout(session))

    def test_sessions_without_a_repository_have_no_git_tools(self):
        agent = A.Agent(console=False, store=A.MemoryStore())
        self.addCleanup(agent.close)
        session = agent.sessions.create()
        self.assertIsNone(run(agent.prepare_session(session)))
        names = {
            tool.name
            for tool in agent.build_tools(A.Workspace(session.workspace), agent.config)
        }
        self.assertNotIn("git_commit", names)

    def test_repository_sessions_expose_the_git_tools(self):
        session = self.agent.sessions.create()
        checkout = run(self.agent.prepare_session(session))
        tools = {
            tool.name: tool
            for tool in self.agent.build_tools(
                A.Workspace(session.workspace), self.agent.config, checkout
            )
        }
        self.assertLessEqual(
            {"git_status", "git_commit", "git_push", "open_pull_request", "publish_work"},
            set(tools),
        )
        (checkout.path / "note.txt").write_text("work", encoding="utf-8")
        answer = ToolRegistryTest.invoke(
            tools["publish_work"], json.dumps({"message": "add note"})
        )
        self.assertIn("https://example/pull/3", answer)

    def test_git_tool_errors_are_reported_to_the_leader(self):
        session = self.agent.sessions.create()
        checkout = run(self.agent.prepare_session(session))
        tools = {
            tool.name: tool
            for tool in self.agent.build_tools(
                A.Workspace(session.workspace), self.agent.config, checkout
            )
        }
        answer = ToolRegistryTest.invoke(tools["git_commit"], json.dumps({"message": "x"}))
        self.assertTrue(answer.startswith("Error:"), answer)

    def test_repo_commands_drive_the_checkout(self):
        created = run(self.agent.commands.invoke("/new"))
        session_id = created["session"]["id"]
        checkout = self.agent.checkout(self.agent.sessions.get(session_id))
        (checkout.path / "note.txt").write_text("work", encoding="utf-8")
        invoke = lambda text: run(
            self.agent.commands.invoke(text, session_id=session_id)
        )
        self.assertIn("note.txt", invoke("/status")["message"])
        self.assertTrue(invoke("/commit add note")["ok"])
        self.assertTrue(invoke("/push")["ok"])
        pull = invoke("/pr Add note\nA body")
        self.assertIn("https://example/pull/3", pull["message"])

    def test_repo_command_shows_and_clears_the_default_repository(self):
        self.assertIn(str(self.origin), run(self.agent.commands.invoke("/repo"))["message"])
        run(self.agent.commands.invoke("/repo owner/name"))
        self.assertEqual(self.agent.repos.spec.slug, "owner/name")
        run(self.agent.commands.invoke("/repo none"))
        self.assertFalse(self.agent.repos.configured)
        self.assertFalse(run(self.agent.commands.invoke("/clone"))["ok"])

    def test_commands_without_a_checkout_report_it(self):
        agent = A.Agent(console=False, store=A.MemoryStore())
        self.addCleanup(agent.close)
        session = agent.sessions.create()
        result = run(agent.commands.invoke("/push", session_id=session.id))
        self.assertFalse(result["ok"])
        self.assertIn("no git checkout", result["message"])


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.commands = A.CommandRegistry()

    def test_parse(self):
        self.assertEqual(A.CommandRegistry.parse("/new  abc "), ("new", "abc"))
        self.assertEqual(A.CommandRegistry.parse("/HELP"), ("help", ""))
        self.assertIsNone(A.CommandRegistry.parse("hello"))
        self.assertIsNone(A.CommandRegistry.parse("/ nope"))

    def test_invoke_sync_and_async_handlers(self):
        self.commands.register("echo", "echo", lambda args, **_: {"ok": True, "args": args})

        async def slow(args, **_):
            return {"ok": True, "args": args.upper()}

        self.commands.register("loud", "loud", slow)
        self.assertEqual(run(self.commands.invoke("/echo hi"))["args"], "hi")
        self.assertEqual(run(self.commands.invoke("/loud hi"))["args"], "HI")
        self.assertIsNone(run(self.commands.invoke("plain text")))
        self.assertFalse(run(self.commands.invoke("/nope"))["ok"])

    def test_describe_is_sorted(self):
        self.commands.register("b", "second", lambda a, **_: None)
        self.commands.register("a", "first", lambda a, **_: None)
        self.assertEqual([c["name"] for c in self.commands.describe()], ["a", "b"])


class AgentCommandTest(unittest.TestCase):
    def setUp(self):
        self.agent = A.Agent(console=False, store=A.MemoryStore())
        self.addCleanup(self.agent.close)

    def test_default_commands_manage_sessions(self):
        created = run(self.agent.commands.invoke("/new"))
        session_id = created["session"]["id"]
        listed = run(self.agent.commands.invoke("/sessions"))
        self.assertIn(session_id, [s["id"] for s in listed["sessions"]])
        self.assertTrue(run(self.agent.commands.invoke(f"/use {session_id}"))["ok"])
        self.assertTrue(run(self.agent.commands.invoke(f"/end {session_id}"))["ok"])
        self.assertFalse(run(self.agent.commands.invoke(f"/use {session_id}"))["ok"])

    def test_model_command_updates_config(self):
        run(self.agent.commands.invoke("/model my-model"))
        self.assertEqual(self.agent.config.model, "my-model")

    def test_vision_command_sets_and_clears_the_vision_model(self):
        self.assertIn("(leader)", run(self.agent.commands.invoke("/vision"))["message"])
        run(self.agent.commands.invoke("/vision my-eyes"))
        self.assertEqual(self.agent.config.vision_model, "my-eyes")
        run(self.agent.commands.invoke("/vision none"))
        self.assertEqual(self.agent.config.vision_model, "")

    def test_build_vision_delegates_to_the_pool(self):
        calls: list[tuple] = []

        class Pool(A.ModelPool):
            async def describe_image(self, spec, data_url, question="", **kwargs):
                calls.append((spec, data_url, question, kwargs))
                return "described"

        agent = A.Agent(
            console=False, store=A.MemoryStore(), models=Pool(), vision_model="v"
        )
        self.addCleanup(agent.close)
        self.assertIsNone(agent.build_vision(A.AgentConfig()))
        delegate = agent.build_vision(agent.config)
        self.assertEqual(run(delegate("data:image/png;base64,AA", "q")), "described")
        spec, data_url, question, kwargs = calls[0]
        self.assertEqual((spec.name, question), ("v", "q"))
        self.assertEqual(kwargs["max_tokens"], agent.config.vision_max_tokens)

    def test_help_lists_commands(self):
        names = [c["name"] for c in run(self.agent.commands.invoke("/help"))["commands"]]
        self.assertIn("help", names)
        self.assertIn("new", names)


class EngineTest(unittest.TestCase):
    """The embeddable loop: no sessions, no storage, no memory, no console."""

    class FakeEngine(A.Engine):
        def build_sdk_agent(self, config, tools):
            self.built = {"config": config, "tools": tools}
            return self.built

        async def stream(self, sdk_agent, model_input, config, result):
            self.inputs = getattr(self, "inputs", [])
            self.inputs.append(model_input)
            result.output = A.prompt_of(model_input).upper()
            return result

    def test_engine_owns_no_batteries(self):
        engine = A.Engine(env=False)
        self.addCleanup(engine.close)
        for battery in ("store", "cache", "sessions", "memory", "repos", "commands"):
            self.assertFalse(hasattr(engine, battery), battery)
        self.assertIsNone(engine.renderer)

    def test_env_layer_can_be_skipped(self):
        import os

        os.environ["AGENT_MODEL"] = "env-model"
        self.addCleanup(os.environ.pop, "AGENT_MODEL", None)
        self.assertEqual(A.Engine(env=False).config.model, A.DEFAULT_MODEL)
        self.assertEqual(A.Engine().config.model, "env-model")

    def test_run_takes_the_model_input_verbatim(self):
        engine = self.FakeEngine(env=False)
        self.addCleanup(engine.close)
        seen: list[str] = []
        engine.events.on(A.EventType.ALL, lambda e: seen.append(e.type))
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = run(engine.run(messages))
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "THIRD")
        self.assertEqual(result.prompt, "third")
        self.assertEqual(engine.inputs[-1], messages)
        self.assertEqual(seen, ["agent.start", "agent.end"])

    def test_run_has_no_tools_without_a_workspace(self):
        engine = self.FakeEngine(env=False)
        self.addCleanup(engine.close)
        run(engine.run("hello"))
        self.assertEqual(engine.built["tools"], [])

    def test_a_workspace_brings_the_built_in_tools(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, True))
        engine = self.FakeEngine(env=False, workspace=root)
        self.addCleanup(engine.close)
        run(engine.run("hello"))
        names = [tool.name for tool in engine.built["tools"]]
        self.assertIn("read_text_file", names)

    def test_tools_may_be_supplied_by_the_consumer(self):
        engine = self.FakeEngine(env=False)
        self.addCleanup(engine.close)
        run(engine.run("hello", tools=["mine"]))
        self.assertEqual(engine.built["tools"], ["mine"])

    def test_run_reports_errors_instead_of_raising(self):
        class Boom(self.FakeEngine):
            async def stream(self, sdk_agent, model_input, config, result):
                raise RuntimeError("boom")

        engine = Boom(env=False)
        self.addCleanup(engine.close)
        result = run(engine.run("hello"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "boom")

    def test_overrides_are_merged_per_run(self):
        engine = self.FakeEngine(env=False)
        self.addCleanup(engine.close)
        run(engine.run("hello", instructions="be brief"))
        self.assertEqual(engine.built["config"].instructions, "be brief")
        self.assertEqual(engine.config.instructions, A.DEFAULT_INSTRUCTIONS)

    def test_agent_is_an_engine(self):
        self.assertTrue(issubclass(A.Agent, A.Engine))


class ToolCallTest(unittest.TestCase):
    """Tool calls are published with their name, arguments, outcome and time."""

    def setUp(self):
        self.engine = A.Engine(env=False)
        self.addCleanup(self.engine.close)

    @staticmethod
    def call_item(name="read_text_file", arguments='{"path": "a.txt"}', call_id="c1"):
        raw = types.SimpleNamespace(name=name, arguments=arguments, call_id=call_id)
        return types.SimpleNamespace(raw_item=raw, tool_name=name)

    @staticmethod
    def output_item(output="contents", call_id="c1"):
        return types.SimpleNamespace(
            raw_item={"call_id": call_id, "output": output},
            output=output,
            call_id=call_id,
        )

    def test_secrets_are_redacted_and_long_values_are_cut(self):
        redacted = A.redact({"path": "a.txt", "api_key": "sk-1", "n": 2, "big": "x" * 20}, 10)
        self.assertEqual(redacted["path"], "a.txt")
        self.assertEqual(redacted["api_key"], A.REDACTED)
        self.assertEqual(redacted["n"], 2)
        self.assertEqual(redacted["big"], "x" * 10 + "\u2026")
        self.assertEqual(A.redact([{"token": "t"}]), [{"token": A.REDACTED}])

    def test_arguments_are_parsed_from_the_json_of_the_model(self):
        self.assertEqual(self.engine.redact_arguments('{"path": "a"}'), {"path": "a"})
        self.assertEqual(self.engine.redact_arguments(""), {})
        self.assertEqual(self.engine.redact_arguments(None), {})
        self.assertEqual(self.engine.redact_arguments("not json"), "not json")

    def test_a_call_reports_its_signature_then_its_outcome(self):
        call = self.engine.tool_call_of(self.call_item(), "b0")
        self.assertEqual(call.name, "read_text_file")
        self.assertEqual(call.call_id, "c1")
        self.assertEqual(call.signature(), 'read_text_file(path="a.txt")')
        self.assertFalse(call.done)
        call.finish("contents")
        self.assertTrue(call.done)
        self.assertIn("-> ok in", call.report())
        self.assertIn("contents", call.report())
        self.assertGreaterEqual(call.duration, 0.0)
        self.assertEqual(call.to_dict()["name"], "read_text_file")

    def test_an_error_result_marks_the_call_as_failed(self):
        self.assertEqual(self.engine.tool_result_of(self.output_item("done")), ("done", True))
        text, ok = self.engine.tool_result_of(self.output_item("Error: nope"))
        self.assertEqual((text, ok), ("Error: nope", False))

    def test_an_output_is_matched_to_its_call(self):
        result = A.RunResult()
        first = A.ToolCall(id="b0", name="a", call_id="c1")
        second = A.ToolCall(id="b1", name="b", call_id="c2")
        result.tools.extend([first, second])
        self.assertIs(A.Engine.pending_tool(result, "c2"), second)
        self.assertIs(A.Engine.pending_tool(result, None), first)
        first.finish("x")
        second.finish("y")
        self.assertIsNone(A.Engine.pending_tool(result, "missing"))

    def test_the_stream_publishes_a_call_from_start_to_end(self):
        from agents import RunItemStreamEvent

        events = [
            RunItemStreamEvent(name="tool_called", item=self.call_item()),
            RunItemStreamEvent(name="tool_output", item=self.output_item()),
        ]

        class Streamed:
            final_output = "done"

            async def stream_events(self):
                for event in events:
                    yield event

        seen = []
        self.engine.events.on(A.EventType.ALL, lambda e: seen.append(e))
        result = A.RunResult(session_id="s1")
        with unittest.mock.patch("agents.Runner.run_streamed", return_value=Streamed()):
            run(self.engine.stream(None, "hi", self.engine.config, result))

        self.assertEqual([e.type for e in seen], ["tool.start", "tool.end"])
        start, end = (e.data for e in seen)
        self.assertEqual(start["name"], "read_text_file")
        self.assertEqual(start["arguments"], {"path": "a.txt"})
        self.assertEqual(start["kind"], "tool")
        self.assertEqual(end["result"], "contents")
        self.assertTrue(end["ok"])
        self.assertGreaterEqual(end["duration"], 0.0)
        self.assertEqual(end["id"], start["id"])
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.text_of("tool"), result.tools[0].report())
        self.assertEqual(result.output, "done")


class AgentRunTest(unittest.TestCase):
    """Exercises the run pipeline with the SDK call stubbed out."""

    class FakeAgent(A.Agent):
        def build_sdk_agent(self, config, tools):
            return {"config": config, "tools": tools}

        async def stream(self, sdk_agent, model_input, config, result):
            self.inputs = getattr(self, "inputs", [])
            self.inputs.append(model_input)
            prompt = A.prompt_of(model_input)
            session_id = result.session_id
            block = A.Block(id="b0", kind="output", session_id=session_id)
            result.blocks.append(block)
            await self.events.publish(
                A.EventType.BLOCK_START, session_id=session_id, id=block.id, kind="output"
            )
            block.text = prompt.upper()
            await self.events.publish(
                A.EventType.BLOCK_DELTA,
                session_id=session_id,
                id=block.id,
                kind="output",
                text=block.text,
            )
            result.output = block.text
            return result

    def setUp(self):
        self.agent = self.FakeAgent(console=False, store=A.MemoryStore())
        self.addCleanup(self.agent.close)

    def test_run_renders_template_and_publishes_events(self):
        seen: list[str] = []
        self.agent.events.on(A.EventType.ALL, lambda e: seen.append(e.type))
        result = run(self.agent.run("Say: {{ config.input }}", input="hello"))
        self.assertTrue(result.ok)
        self.assertEqual(result.prompt, "Say: hello")
        self.assertEqual(result.output, "SAY: HELLO")
        self.assertEqual(result.text_of("output"), "SAY: HELLO")
        self.assertEqual(
            seen, ["agent.start", "block.start", "block.delta", "agent.end"]
        )

    def test_run_reuses_a_given_session(self):
        session = self.agent.sessions.create()
        result = run(self.agent.run("{{ config.input }}", session=session, input="x"))
        self.assertEqual(result.session_id, session.id)

    def test_run_reports_errors_instead_of_raising(self):
        result = run(self.agent.run("{{ missing }}", input="x"))
        self.assertFalse(result.ok)
        self.assertIn("missing", result.error)

    def test_attachments_are_appended_to_the_prompt(self):
        result = run(
            self.agent.run("Base", input="x", attachments=["attachments/a.png"])
        )
        self.assertIn("## Attached files", result.prompt)
        self.assertIn("attachments/a.png", result.prompt)

    def test_close_wipes_every_workspace(self):
        result = run(self.agent.run("{{ config.input }}", input="x"))
        workspace = self.agent.sessions.get(result.session_id).workspace
        self.agent.close()
        self.assertFalse(workspace.exists())

    def test_durable_sessions_survive_a_restart(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, True))
        db = root / "agent.db"
        first = self.FakeAgent(
            console=False,
            store=A.SqliteStore(db),
            workspace_root=root,
            session_durable=True,
        )
        result = run(first.run("{{ config.input }}", input="hello"))
        workspace = first.sessions.get(result.session_id).workspace
        first.close()
        self.assertTrue(workspace.is_dir())
        second = self.FakeAgent(
            console=False,
            store=A.SqliteStore(db),
            workspace_root=root,
            session_durable=True,
        )
        self.addCleanup(lambda: second.sessions.close_all(destroy=True))
        restored = second.sessions.get(result.session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.workspace, workspace)
        self.assertEqual(
            second.memory.history(result.session_id),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "HELLO"},
            ],
        )

    def test_a_second_run_replays_the_conversation(self):
        session = self.agent.sessions.create()
        run(self.agent.run("{{ config.input }}", session=session, input="hello"))
        run(self.agent.run("{{ config.input }}", session=session, input="again"))
        self.assertEqual(self.agent.inputs[0], "hello")
        self.assertEqual(
            self.agent.inputs[1],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "HELLO"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_the_input_is_remembered_rather_than_the_rendered_prompt(self):
        session = self.agent.sessions.create()
        run(self.agent.run("Say: {{ config.input }}", session=session, input="hello"))
        self.assertEqual(
            self.agent.memory.history(session.id),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "SAY: HELLO"},
            ],
        )

    def test_a_failed_run_is_not_remembered(self):
        result = run(self.agent.run("{{ missing }}", input="x"))
        self.assertEqual(self.agent.memory.history(result.session_id), [])

    def test_memory_can_be_disabled(self):
        agent = self.FakeAgent(console=False, store=A.MemoryStore(), memory_enabled=False)
        self.addCleanup(agent.close)
        session = agent.sessions.create()
        run(agent.run("{{ config.input }}", session=session, input="hello"))
        run(agent.run("{{ config.input }}", session=session, input="again"))
        self.assertEqual(agent.inputs, ["hello", "again"])
        self.assertEqual(agent.memory.history(session.id), [])

    def test_forget_clears_the_conversation_of_a_session(self):
        session = self.agent.sessions.create()
        run(self.agent.run("{{ config.input }}", session=session, input="hello"))
        result = run(self.agent.commands.invoke("/forget", session_id=session.id))
        self.assertTrue(result["ok"])
        self.assertEqual(self.agent.memory.history(session.id), [])

    def test_ending_a_session_forgets_it(self):
        session = self.agent.sessions.create()
        run(self.agent.run("{{ config.input }}", session=session, input="hello"))
        run(self.agent.commands.invoke("/end", session_id=session.id))
        self.assertEqual(self.agent.memory.history(session.id), [])

    def test_a_summarizer_is_only_built_when_asked_for(self):
        self.assertIsNone(self.agent.memory.summarizer)
        agent = self.FakeAgent(
            console=False, store=A.MemoryStore(), memory_summary=True, model="m"
        )
        self.addCleanup(agent.close)
        self.assertIsNotNone(agent.memory.summarizer)


class ConsoleRendererTest(unittest.TestCase):
    class Buffer:
        def __init__(self):
            self.text = ""

        def write(self, chunk):
            self.text += chunk

        def flush(self):
            pass

        def isatty(self):
            return False

    def setUp(self):
        self.buffer = self.Buffer()
        self.renderer = A.ConsoleRenderer(color=False, stream=self.buffer)

    def test_block_changes_insert_one_blank_line(self):
        self.renderer.emit("reasoning", "thinking")
        self.renderer.emit("output", "answer")
        self.assertEqual(self.buffer.text, "thinking\n\nanswer")

    def test_existing_newlines_are_not_doubled(self):
        self.renderer.emit("output", "line\n\n")
        self.renderer.emit("reasoning", "\n\nnext")
        self.assertEqual(self.buffer.text, "line\n\nnext")

    def test_empty_text_is_ignored(self):
        self.renderer.emit("output", "")
        self.assertEqual(self.buffer.text, "")

    def test_color_styles_are_applied_when_enabled(self):
        renderer = A.ConsoleRenderer(color=True, stream=self.Buffer())
        self.assertTrue(renderer.style("reasoning", "x").endswith("\x1b[0m"))
        self.assertEqual(A.ConsoleRenderer(color=False, stream=self.buffer).style("x", "y"), "y")

    def test_banner_is_boxed(self):
        self.renderer.banner("title", [("Key", "value")])
        lines = self.buffer.text.splitlines()
        self.assertTrue(lines[0].startswith("+---"))
        self.assertIn("Key -> value", self.buffer.text)

    def test_renderer_subscribes_to_a_bus(self):
        bus = A.EventBus()
        self.renderer.attach(bus)
        run(bus.publish(A.EventType.BLOCK_DELTA, kind="output", text="hi"))
        self.assertEqual(self.buffer.text, "hi")

    def test_tool_calls_are_printed_with_their_outcome(self):
        bus = A.EventBus()
        self.renderer.attach(bus)
        run(bus.publish(A.EventType.TOOL_START, kind="tool", text='read(path="a")'))
        run(
            bus.publish(
                A.EventType.TOOL_END,
                kind="tool",
                name="read",
                ok=False,
                duration=0.002,
                result="Error: nope",
            )
        )
        self.assertIn('-> read(path="a")', self.buffer.text)
        self.assertIn("<- read: failed in 2 ms", self.buffer.text)
        self.assertIn("Error: nope", self.buffer.text)

    def test_truncate(self):
        self.assertEqual(A.truncate("abcdef", 10), "abcdef")
        self.assertEqual(A.truncate("abcdefghij", 6), "abc...")
        self.assertEqual(A.truncate("/a/very/long/path.txt", 10), "...ath.txt")


class ThemeTest(unittest.TestCase):
    def test_vscode_theme_maps_onto_css_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theme.json"
            path.write_text(
                json.dumps(
                    {
                        "type": "dark",
                        "colors": {
                            "editor.background": "#101010",
                            "editor.foreground": "#f0f0f0",
                            "unsupported.key": "#123456",
                        },
                    }
                ),
                encoding="utf-8",
            )
            theme = A.load_vscode_theme(path)
        self.assertEqual(theme["--bg"], "#101010")
        self.assertEqual(theme["--fg"], "#f0f0f0")
        self.assertEqual(theme["--theme-type"], "dark")
        self.assertNotIn("unsupported.key", theme)

    def test_missing_theme_is_empty(self):
        self.assertEqual(A.load_vscode_theme(None), {})


class HubTest(unittest.TestCase):
    class FakeSocket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    def setUp(self):
        self.hub = A.Hub()
        self.socket = self.FakeSocket()
        self.connection = A.Connection(self.hub, self.socket)
        self.hub.connections[self.connection.id] = self.connection

    def test_channels_receive_broadcasts(self):
        self.hub.join(self.connection, "room")
        run(self.hub.broadcast("room", "tick", n=1))
        self.hub.leave(self.connection, "room")
        run(self.hub.broadcast("room", "tick", n=2))
        self.assertEqual(self.socket.sent, [{"type": "tick", "n": 1}])

    def test_dispatch_routes_to_handlers(self):
        seen = []
        self.hub.on("hi", lambda conn, msg: seen.append(msg["v"]))
        run(self.hub.dispatch(self.connection, {"type": "hi", "v": 7}))
        self.assertEqual(seen, [7])

    def test_dispatch_reports_unknown_types(self):
        run(self.hub.dispatch(self.connection, {"type": "nope"}))
        self.assertEqual(self.socket.sent[0]["type"], "error")


class WebServerTest(unittest.TestCase):
    def setUp(self):
        self.agent = A.Agent(console=False, store=A.MemoryStore())
        self.addCleanup(self.agent.close)
        self.server = A.WebServer(self.agent, template="{{ config.input }}")

    def test_rest_endpoints(self):
        status, body = self.server.rest("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertIn("model", json.loads(self.server.rest("/api/config")[1]))
        self.assertIsInstance(json.loads(self.server.rest("/api/commands")[1]), list)
        self.assertIsNone(self.server.rest("/api/unknown"))

    def test_event_payloads_drop_non_serializable_values(self):
        event = A.Event(
            type=A.EventType.BLOCK_DELTA, data={"text": "x", "config": A.AgentConfig()}
        )
        self.assertEqual(self.server.encode(event), {"text": "x"})

    def test_bind_moves_a_connection_between_channels(self):
        connection = A.Connection(self.server.hub, HubTest.FakeSocket())
        self.server.hub.connections[connection.id] = connection
        self.server.bind(connection, "one")
        self.server.bind(connection, "two")
        self.assertEqual(connection.channels, {A.WebServer.channel("two")})

    def test_hello_replays_the_conversation_of_the_session(self):
        session = self.agent.sessions.create()
        run(
            self.agent.memory.remember(
                session.id, [A.Turn("user", "hello"), A.Turn("assistant", "hi")]
            )
        )
        connection = A.Connection(self.server.hub, HubTest.FakeSocket())
        self.server.hub.connections[connection.id] = connection
        run(self.server.on_hello(connection, {"session": session.id}))
        hello = connection.ws.sent[-1]
        self.assertEqual(hello["session"]["id"], session.id)
        self.assertEqual(
            [(t["role"], t["text"]) for t in hello["history"]],
            [("user", "hello"), ("assistant", "hi")],
        )

    def test_attachments_are_stored_inside_the_session(self):
        session = self.agent.sessions.create()
        saved = self.server.store_attachments(
            session,
            [{"name": "../../evil name.png", "data": base64.b64encode(b"png").decode()}],
        )
        self.assertEqual(saved, ["attachments/evil_name.png"])
        self.assertEqual((session.workspace / saved[0]).read_bytes(), b"png")

    def test_oversized_attachments_are_rejected(self):
        session = self.agent.sessions.create()
        blob = base64.b64encode(b"x" * (A.MAX_ATTACHMENT_BYTES + 1)).decode()
        with self.assertRaises(ValueError):
            self.server.store_attachments(session, [{"name": "big.bin", "data": blob}])

    def test_invalid_attachment_encoding_is_rejected(self):
        session = self.agent.sessions.create()
        with self.assertRaises(ValueError):
            self.server.store_attachments(session, [{"name": "a.bin", "data": "@@@"}])

    def test_safe_name(self):
        self.assertEqual(A.safe_name("../../etc/passwd"), "passwd")
        self.assertEqual(A.safe_name(""), "file")
        self.assertEqual(A.safe_name("a b?c.txt"), "a_b_c.txt")


class CliTest(unittest.TestCase):
    def test_only_input_and_serve_are_accepted(self):
        args = A.parse_args(["--input", "notes.md", "--serve"])
        self.assertEqual(args.input, "notes.md")
        self.assertTrue(args.serve)
        self.assertEqual(vars(A.parse_args([])), {"input": "", "serve": False})
        with self.assertRaises(SystemExit):
            A.parse_args(["--unknown"])

    def test_template_resolution_prefers_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.md"
            path.write_text("from file", encoding="utf-8")
            import os

            os.environ["AGENT_TEMPLATE"] = str(path)
            self.addCleanup(os.environ.pop, "AGENT_TEMPLATE", None)
            self.assertEqual(A.load_template(), "from file")
            os.environ["AGENT_TEMPLATE"] = "raw {{ config.input }}"
            self.assertEqual(A.load_template(), "raw {{ config.input }}")


if __name__ == "__main__":
    unittest.main()
