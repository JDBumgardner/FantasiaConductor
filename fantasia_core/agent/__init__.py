"""Agent layer — exposes the CommandBus + project queries as Claude tools.

The agent dispatches the *same* Command objects the UI does, so everything it
does is undoable and consistent. This package is Qt-free; the UI drives it.
"""

from fantasia_core.agent.tools import AgentTools
from fantasia_core.agent.session import AgentSession, DEFAULT_MODEL

__all__ = ["AgentTools", "AgentSession", "DEFAULT_MODEL"]
