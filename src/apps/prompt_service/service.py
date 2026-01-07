from apps.prompt_service.models import Prompt
from common.errors import PromptDataError, PromptNotFoundError
from common.types import PromptContext, SystemPrompt


class PromptService:
    def select_system_prompt(self, context: PromptContext) -> SystemPrompt:
        del context
        active_prompts = Prompt.objects.filter(is_active=True).order_by(
            '-version', '-created_at', '-id'
        )
        latest = active_prompts.first()
        if not latest:
            raise PromptNotFoundError("No active prompt versions found.")

        conflicts = active_prompts.filter(version=latest.version).exclude(id=latest.id)
        if conflicts.exists():
            raise PromptDataError(
                "Multiple active prompt versions share the latest version value."
            )

        return SystemPrompt(
            prompt_id=latest.id,
            version=latest.version,
            text=latest.text,
        )
