from django.test import TestCase

from apps.evaluation_service.service import EvaluationService
from apps.history_service.models import Conversation, Turn
from apps.prompt_service.models import Prompt
from apps.ratings_service.models import Evaluator, TurnFeedback


class EvaluationServiceTests(TestCase):
    def setUp(self):
        self.prompt = Prompt.objects.create(text='Prompt A', is_active=True)
        self.evaluator = Evaluator.objects.create(
            name='qscore_v2',
            version='2.0.0',
            config_json={
                'weights': {
                    'w_progress': 0.4737,
                    'w_confusion_reduction': 0.2632,
                    'w_clarity': 0.1579,
                    'w_engagement': 0.1053,
                },
                'rating_scale': {'min': 1, 'max': 5},
            },
        )

    def test_first_rated_turn_uses_neutral_deltas(self):
        conversation = Conversation.objects.create(user_id='learner-1')
        turn = Turn.objects.create(
            conversation=conversation,
            turn_index=0,
            user_text='u0',
            assistant_text='a0',
            prompt=self.prompt,
        )
        TurnFeedback.objects.create(
            turn=turn,
            user_id='learner-1',
            rating_perceived_progress=5,
            rating_clarity_understanding=4,
            rating_engagement_fit=3,
        )

        evaluation = EvaluationService().evaluate_turn(turn)

        self.assertIsNotNone(evaluation)
        self.assertAlmostEqual(evaluation.q_progress, 0.5, places=6)
        self.assertAlmostEqual(evaluation.q_confusion_reduction, 0.5, places=6)
        self.assertAlmostEqual(evaluation.q_clarity, 0.75, places=6)
        self.assertAlmostEqual(evaluation.q_engagement, 0.5, places=6)
        expected_total = (0.4737 * 0.5) + (0.2632 * 0.5) + (0.1579 * 0.75) + (0.1053 * 0.5)
        self.assertAlmostEqual(evaluation.q_total, expected_total, places=6)

    def test_follow_up_turn_uses_delta_scores(self):
        conversation = Conversation.objects.create(user_id='learner-1')

        turn0 = Turn.objects.create(
            conversation=conversation,
            turn_index=0,
            user_text='u0',
            assistant_text='a0',
            prompt=self.prompt,
        )
        TurnFeedback.objects.create(
            turn=turn0,
            user_id='learner-1',
            rating_perceived_progress=2,
            rating_clarity_understanding=2,
            rating_engagement_fit=3,
        )
        EvaluationService().evaluate_turn(turn0)

        turn1 = Turn.objects.create(
            conversation=conversation,
            turn_index=1,
            user_text='u1',
            assistant_text='a1',
            prompt=self.prompt,
        )
        TurnFeedback.objects.create(
            turn=turn1,
            user_id='learner-1',
            rating_perceived_progress=4,
            rating_clarity_understanding=5,
            rating_engagement_fit=4,
        )

        evaluation = EvaluationService().evaluate_turn(turn1)

        self.assertIsNotNone(evaluation)
        self.assertAlmostEqual(evaluation.q_progress, 0.75, places=6)
        self.assertAlmostEqual(evaluation.q_confusion_reduction, 0.875, places=6)
        self.assertAlmostEqual(evaluation.q_clarity, 1.0, places=6)
        self.assertAlmostEqual(evaluation.q_engagement, 0.75, places=6)
        expected_total = (0.4737 * 0.75) + (0.2632 * 0.875) + (0.1579 * 1.0) + (0.1053 * 0.75)
        self.assertAlmostEqual(evaluation.q_total, expected_total, places=6)
