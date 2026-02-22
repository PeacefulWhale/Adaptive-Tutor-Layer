import uuid

from django.test import TestCase

from apps.handler.service import TutorResponseHandler
from apps.prompt_service.models import Prompt, PromptDecision


class FakeLLMService:
    def generate(self, system_prompt_text: str, history_turns: list, user_message: str) -> dict:
        return {
            'assistant_text': f"reply to {user_message}",
            'metadata': {'source': 'fake-llm'},
        }


class HandlerBanditLinkingTests(TestCase):
    def test_turn_linking_uses_conversation_learner_and_turn_number(self):
        prompt = Prompt.objects.create(text='Prompt A', is_active=True)
        conversation_id = str(uuid.uuid4())

        # Same conversation + turn_number but different learner should remain unlinked.
        other_decision = PromptDecision.objects.create(
            learner_id='other-learner',
            conversation_id=conversation_id,
            prompt=prompt,
            turn_number=0,
            sampled_theta=0.1,
            model_version='pts_normal_v1',
        )

        handler = TutorResponseHandler(llm_service=FakeLLMService())
        result = handler.generate_response(
            user_id='learner-1',
            conversation_id=conversation_id,
            question_text='What is a gradient?',
        )

        learner_decision = PromptDecision.objects.get(
            learner_id='learner-1',
            conversation_id=conversation_id,
            turn_number=0,
        )
        other_decision.refresh_from_db()

        self.assertEqual(str(learner_decision.turn_id), result['turn_id'])
        self.assertIsNone(other_decision.turn_id)
