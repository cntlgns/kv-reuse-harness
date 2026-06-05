import os
from jinja2 import Template
import logging
from typing import Dict, Optional
import yaml


LOG = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(
        self,
        base_dir: Optional[str] = None,
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))  # the dir for the base prompt
        self.prompt_temp_dict: Dict[str, Dict] = {}
        # find all yaml file in base_dir (generic prompts, e.g, system/user-prompts)
        for file in os.listdir(self.base_dir):
            if file.endswith(".yaml"):
                with open(os.path.join(self.base_dir, file), "r") as f:
                    self.prompt_temp_dict[file[:-5]] = yaml.safe_load(f)
                LOG.info(f"Loaded prompt template: {file}")

    def get_prompt(self, agent_name: str, templates_name: str, **kwargs) -> str:
        template = Template(self.prompt_temp_dict[agent_name][templates_name])
        return template.render(**kwargs)

    def get_system_prompt(self, agent_name: str, **kwargs) -> str:
        prompt_tag = kwargs.get("prompt_tag", "")
        return self.get_prompt(agent_name, f"system_template_{prompt_tag}", **kwargs)

    def get_user_prompt(self, agent_name: str, **kwargs) -> str:
        prompt_tag = kwargs.get("prompt_tag", "")
        return self.get_prompt(agent_name, f"user_template_{prompt_tag}", **kwargs)
