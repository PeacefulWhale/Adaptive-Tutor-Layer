import json
import uuid
from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from apps.prompt_service.models import Prompt, PromptDecision


class RewardVarianceCommandTests(TestCase):
    def _create_decision(self, prompt: Prompt, reward_version: str, reward: float, learner_id: str = 'learner-1'):
        PromptDecision.objects.create(
            learner_id=learner_id,
            conversation_id=uuid.uuid4(),
            prompt=prompt,
            turn_number=0,
            sampled_theta=0.0,
            model_version='pts_normal_v1',
            reward=reward,
            reward_version=reward_version,
        )

    def test_reports_improvement_when_candidate_variance_is_higher(self):
        prompt = Prompt.objects.create(text='Prompt A', is_active=True)

        for value in [0.40, 0.50, 0.45, 0.55]:
            self._create_decision(prompt, 'mean_q_v0', value)
        for value in [0.10, 0.90, 0.20, 0.80]:
            self._create_decision(prompt, 'q_v2', value)

        out = StringIO()
        call_command(
            'analyze_reward_variance',
            '--baseline-version',
            'mean_q_v0',
            '--candidate-version',
            'q_v2',
            stdout=out,
        )
        payload = json.loads(out.getvalue())

        self.assertEqual(payload['baseline']['count'], 4)
        self.assertEqual(payload['candidate']['count'], 4)
        self.assertTrue(payload['improved'])
        self.assertGreater(payload['variance_delta'], 0.0)

    def test_fail_if_not_improved_raises_command_error(self):
        prompt = Prompt.objects.create(text='Prompt A', is_active=True)

        for value in [0.10, 0.90, 0.20, 0.80]:
            self._create_decision(prompt, 'mean_q_v0', value)
        for value in [0.45, 0.50, 0.48, 0.52]:
            self._create_decision(prompt, 'q_v2', value)

        out = StringIO()
        with self.assertRaises(CommandError):
            call_command(
                'analyze_reward_variance',
                '--baseline-version',
                'mean_q_v0',
                '--candidate-version',
                'q_v2',
                '--fail-if-not-improved',
                stdout=out,
            )
