#!/usr/bin/env python3
"""ICE Terminal UI – full-featured memory-aware chat client."""

import asyncio, httpx, json, uuid, os, sys
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Input, RichLog, TabbedContent, TabPane,
    Button, Static, Select, TextArea
)
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

ICE_PROXY = "http://localhost:8000"
CHAT_URL = f"{ICE_PROXY}/v1/chat/completions"
USER_CONTROL = f"{ICE_PROXY}/user-control"
MEMORY_SLOTS = f"{ICE_PROXY}/memory-slots"

SCOPES = [
    ("auto", "auto"),
    ("project", "project"),
    ("none", "none"),
]

class ICETUI(App):
    CSS = """
    TabbedContent { height: 1fr; }
    #context_info { height: auto; padding: 1; }
    """

    BINDINGS = [
        Binding("ctrl+b", "bookmark", "Bookmark last turn"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent("Chat", "Context", "Memory", "Settings"):
            # ---- Chat tab ----
            with TabPane("Chat"):
                self.chat_log = RichLog(highlight=True, markup=True)
                yield self.chat_log
                self.chat_input = Input(placeholder="Type your message...")
                yield self.chat_input

            # ---- Context tab ----
            with TabPane("Context"):
                self.context_display = RichLog(highlight=True, markup=True)
                yield self.context_display

            # ---- Memory tab ----
            with TabPane("Memory"):
                self.memory_area = Vertical()
                yield self.memory_area

            # ---- Settings tab ----
            with TabPane("Settings"):
                yield Static("Memory Scope:")
                self.scope_select = Select(SCOPES, value="auto")
                yield self.scope_select
                yield Button("Apply Scope", id="apply_scope")
                self.scope_status = Static("")
                yield self.scope_status

        yield Footer()

    def on_mount(self) -> None:
        global CONV_ID
        CONV_ID = str(uuid.uuid4())
        self.current_scope = "auto"
        self.last_turn_id = None

        self.chat_log.write(f"[bold green]ICE Terminal UI – conversation {CONV_ID[:8]}[/bold green]")
        self.chat_log.write("Type a message and press Enter.\n")

        # Load memory slots
        asyncio.create_task(self.load_memory_slots())
        # Load current scope from backend (if conversation exists)
        asyncio.create_task(self.load_scope())

    # ------------------------------------------------------------------
    # Chat handling
    # ------------------------------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        self.chat_input.value = ""
        self.chat_log.write(f"\n[bold cyan]You:[/bold cyan] {prompt}")
        self.chat_log.write("[dim]ICE is thinking...[/dim]")

        # Clear context display for new turn
        self.context_display.clear()
        self.context_display.write("[bold yellow]Waiting for classifier...[/bold yellow]")

        # Send the request and stream SSE events
        await self.stream_chat(prompt)

    async def stream_chat(self, prompt: str):
        """Stream the chat response and SSE telemetry from the proxy."""
        accumulated_text = ""
        self.last_turn_id = None

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                CHAT_URL,
                json={
                    "model": "ice-proxy",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
                headers={"X-ICE-Conversation-ID": CONV_ID}
            ) as response:
                event_type = None
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if event_type == "classified":
                            data = json.loads(data_str)
                            self.update_context_classified(data)
                        elif event_type == "retrieval":
                            data = json.loads(data_str)
                            self.update_context_retrieval(data)
                        elif event_type == "context_ready":
                            data = json.loads(data_str)
                            self.update_context_ready(data)
                        elif event_type == "generating":
                            data = json.loads(data_str)
                            self.context_display.write(f"[bold]Model:[/bold] {data.get('model', 'unknown')}")
                        else:
                            # Regular LLM token
                            try:
                                token_data = json.loads(data_str)
                                content = token_data["choices"][0]["delta"].get("content", "")
                                if content:
                                    accumulated_text += content
                                    self.chat_log.write(content, end="")
                            except Exception:
                                pass  # ignore non-JSON lines

        # After streaming finishes, display the full response
        self.chat_log.write("\n")
        # Fetch the latest turn ID for bookmarking
        await self.fetch_latest_turn()

    # ------------------------------------------------------------------
    # Context display updates
    # ------------------------------------------------------------------
    def update_context_classified(self, data: dict):
        self.context_display.write(
            f"[bold]Classified:[/bold] {data.get('topic_tags', [])} | "
            f"{data.get('intent_tags', [])} | "
            f"reliance={data.get('context_reliance', '')} | "
            f"confidence={data.get('max_confidence', 0):.2f}"
        )

    def update_context_retrieval(self, data: dict):
        self.context_display.write(
            f"[bold]Retrieval:[/bold] legs={data.get('active_legs', [])}, "
            f"HyDE={data.get('hyde_used', False)}, "
            f"tokens={data.get('tokens_injected', 0)}"
        )

    def update_context_ready(self, data: dict):
        sources = data.get('sources', {})
        self.context_display.write(
            f"[bold]Context ready:[/bold] {data.get('fragments_count', 0)} fragments "
            f"(Codex:{sources.get('codex',0)}, Epi:{sources.get('episodic',0)}, "
            f"Proc:{sources.get('procedural',0)}, RAG:{sources.get('rag',0)}), "
            f"total tokens={data.get('total_tokens', 0)}"
        )

    # ------------------------------------------------------------------
    # Bookmarking
    # ------------------------------------------------------------------
    async def action_bookmark(self) -> None:
        if not self.last_turn_id:
            self.chat_log.write("[red]No turn to bookmark yet.[/red]")
            return
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{USER_CONTROL}/turns/{self.last_turn_id}/bookmark")
            if resp.status_code == 200:
                self.chat_log.write("[bold green]Turn bookmarked![/bold green]")
            else:
                self.chat_log.write(f"[red]Bookmark failed: {resp.status_code}[/red]")

    async def fetch_latest_turn(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{USER_CONTROL}/conversations/{CONV_ID}/latest-turn")
            if resp.status_code == 200:
                data = resp.json()
                self.last_turn_id = data["turn_id"]

    # ------------------------------------------------------------------
    # Memory slots
    # ------------------------------------------------------------------
    async def load_memory_slots(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(MEMORY_SLOTS)
            if resp.status_code == 200:
                slots = resp.json()
                await self.build_memory_ui(slots)

    async def build_memory_ui(self, slots: list):
        await self.memory_area.remove_children()
        for slot in slots:
            slot_name = slot["slot_name"]
            content = slot.get("content", "")
            label = Static(f"[bold]{slot_name}[/bold]")
            text_area = TextArea(content, id=f"slot_{slot_name}")
            save_btn = Button("Save", id=f"save_{slot_name}")
            await self.memory_area.mount(label)
            await self.memory_area.mount(text_area)
            await self.memory_area.mount(save_btn)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id and btn_id.startswith("save_"):
            slot_name = btn_id[5:]
            text_area = self.query_one(f"#slot_{slot_name}", TextArea)
            new_content = text_area.text
            async with httpx.AsyncClient() as client:
                resp = await client.put(
                    f"{MEMORY_SLOTS}/{slot_name}",
                    json={"content": new_content}
                )
                if resp.status_code == 200:
                    self.chat_log.write(f"[green]Slot '{slot_name}' updated.[/green]")
                else:
                    self.chat_log.write(f"[red]Failed to update slot '{slot_name}'[/red]")
        elif btn_id == "apply_scope":
            await self.apply_scope()

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------
    async def load_scope(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{USER_CONTROL}/conversations/{CONV_ID}/scope")
            if resp.status_code == 200:
                data = resp.json()
                self.current_scope = data["memory_scope_type"]
                self.scope_select.value = self.current_scope

    async def apply_scope(self):
        new_scope = self.scope_select.value
        self.current_scope = new_scope
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{USER_CONTROL}/conversations/{CONV_ID}/scope",
                json={"memory_scope_type": new_scope}
            )
            if resp.status_code == 200:
                self.scope_status.update(f"Scope set to [bold]{new_scope}[/bold]")
            else:
                self.scope_status.update(f"[red]Failed to set scope[/red]")


if __name__ == "__main__":
    app = ICETUI()
    app.run()