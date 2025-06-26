import yaml
import os

class PromptManager:
    """Manages loading and versioning of all prompts."""

    def __init__(self, prompts_config_path="config/prompts.yaml", prompt_templates_dir="prompts"):
        self.prompts_config_path = prompts_config_path
        self.prompt_templates_dir = prompt_templates_dir
        self.prompts_config = self._load_prompts_config()

    def _load_prompts_config(self):
        with open(self.prompts_config_path, "r") as f:
            return yaml.safe_load(f)

    def get_prompt(self, task_name):
        """Retrieves the prompt template for a given task."""
        prompt_info = self.prompts_config.get(task_name)
        if not prompt_info:
            raise ValueError(f"No prompt configured for task: {task_name}")

        version = prompt_info.get("version", "default")
        prompt_file = prompt_info.get("file")

        if not prompt_file:
            raise ValueError(f"Prompt file not specified for task: {task_name}")

        # Assuming prompt files are in a 'prompts' directory relative to the project root
        prompt_path = os.path.join(self.prompt_templates_dir, prompt_file)

        with open(prompt_path, "r") as f:
            return f.read()


