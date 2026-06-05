from dataclasses import dataclass
from strands.hooks import HookEvent


@dataclass
class AgentCompletedEvent(HookEvent):
    """Event triggered when an agent has completed execution.

    This event is triggered after the agent has been finished its execution 
    and is either gracefully exiting or terminated due to some unhandled 
    exception. In any case, this event marks the end of agent-call.
    """

    pass