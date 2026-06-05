
import json
import logging
from typing import List
from strands.hooks import MessageAddedEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.content import Message

from .events import AgentCompletedEvent


LOG = logging.getLogger(__name__)


class TrajectoryHook(HookProvider):
    def __init__(
        self,
        output_dir: str,
        log_traj: bool = True,
        record_interval: int = 4,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.log_traj = log_traj
        self.record_interval = record_interval
        self.record_counter = 0

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(MessageAddedEvent, self.dump_trajectory)
        registry.add_callback(AgentCompletedEvent, self.completion)
    
    def _save_traj(self, messages: List[Message]) -> None:
        output_path = f"{self.output_dir}/trajectory.json"
        with open(output_path, "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False, default=str)
    
    def dump_trajectory(self, event: MessageAddedEvent) -> None:
        self.record_counter += 1
        if self.record_counter % self.record_interval != 0:
            return

        messages: List[Message] = event.agent.messages
        if not messages:
            return
        self._save_traj(messages)

    def completion(self, event: AgentCompletedEvent) -> None:
        messages: List[Message] = event.agent.messages
        self._save_traj(messages)
        if self.log_traj:
            for i, msg in enumerate(messages):
                data = json.dumps(msg, indent=4, ensure_ascii=False, default=str)
                LOG.info(f'[{i}]. {data.encode().decode("unicode_escape")}')

