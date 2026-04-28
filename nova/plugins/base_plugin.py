"""AbstractPlugin — base class for all Nova skill plugins.

Every plugin must:
  1. Implement the `supported_intents` property.
  2. Implement the `execute` async method.
  3. Optionally implement `on_load` and `on_unload` lifecycle hooks.

Plugins are auto-discovered by the PluginRegistry via importlib.
They must be placed in a directory listed in `settings.yaml#plugins.plugin_dirs`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class AbstractPlugin(ABC):
    """Abstract base class for Nova skill plugins.

    Subclass this, implement the required methods, and place the module
    in a directory the PluginRegistry scans.

    Example::

        class WeatherPlugin(AbstractPlugin):
            @property
            def name(self) -> str:
                return "weather"

            @property
            def supported_intents(self) -> List[str]:
                return ["get_weather"]

            async def execute(self, intent, context) -> "ExecutionResult":
                ...
    """

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier, e.g. 'weather', 'spotify', 'calculator'."""
        ...

    @property
    @abstractmethod
    def supported_intents(self) -> List[str]:
        """List of intent names this plugin can handle.

        Returns:
            A list of intent_name strings matching IntentObject.intent_name.
        """
        ...

    # ------------------------------------------------------------------
    # Required method
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, intent, context) -> object:
        """Execute the plugin for a matched intent.

        Args:
            intent: IntentObject from the NLU pipeline.
            context: ContextManager instance with session state.

        Returns:
            ExecutionResult with .response_text and optional .undo_data.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    async def on_load(self) -> None:
        """Called once when the plugin is loaded by the PluginRegistry.

        Use this to initialise connections, load models, or set up state.
        """

    async def on_unload(self) -> None:
        """Called once when the plugin is unloaded or Nova shuts down.

        Use this to clean up resources, close connections, etc.
        """

    # ------------------------------------------------------------------
    # Optional introspection
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        """Human-readable description shown in the plugin manager UI."""
        return ""

    @property
    def version(self) -> str:
        """Semantic version string of this plugin, e.g. '1.0.0'."""
        return "0.0.1"

    @property
    def author(self) -> str:
        """Plugin author name."""
        return "Unknown"

    def can_handle(self, intent_name: str) -> bool:
        """Return True if this plugin handles *intent_name*."""
        return intent_name in self.supported_intents

    def __repr__(self) -> str:
        return f"<Plugin name={self.name!r} intents={self.supported_intents!r}>"
