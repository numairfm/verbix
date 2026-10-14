"""AnnouncerAgent: Tool-less voice presenter producing spoken debriefs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from verbix.agent.voice import VoiceAgent, VoiceConfig


class AnnouncerConfig(VoiceConfig):
    """Configuration for AnnouncerAgent."""

    tools: list[str] = Field(default_factory=list)
    tool_choice: str | None = "none"
    spoken_error_fallback: str = ""


class AnnouncerAgent(VoiceAgent):
    """Tool-less voice presenter producing spoken debriefs."""

    def __init__(
        self,
        model: Any,
        tools: Any = None,
        config: AnnouncerConfig | VoiceConfig | BaseModel | dict[str, Any] | None = None,
        env: Any = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            cfg = AnnouncerConfig(**kwargs)
        elif isinstance(config, dict):
            cfg = AnnouncerConfig(**config)
        elif isinstance(config, AnnouncerConfig):
            cfg = config
        elif isinstance(config, BaseModel):
            cfg = AnnouncerConfig(**config.model_dump())
        else:
            cfg = AnnouncerConfig(**kwargs)

        cfg.tools = []
        cfg.tool_choice = "none"
        cfg.spoken_error_fallback = ""
        super().__init__(model=model, tools=None, config=cfg, env=env)

    def get_tool_definitions(
        self, excluded: set[str] | None = None
    ) -> list[dict[str, Any]] | None:
        return None
