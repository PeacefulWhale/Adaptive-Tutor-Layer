from __future__ import annotations

from dataclasses import dataclass

from apps.history_service.models import Turn
from apps.ratings_service.models import Evaluator, TurnEvaluation, TurnFeedback


@dataclass(frozen=True)
class EvaluationRewardPayload:
    evaluator_name: str
    q_total: float
    q_progress: float
    q_confusion_reduction: float
    q_clarity: float
    q_engagement: float


class EvaluationService:
    DEFAULT_EVALUATOR_NAME = 'qscore_v2'

    def __init__(self, evaluator_name: str | None = None):
        self.evaluator_name = evaluator_name or self.DEFAULT_EVALUATOR_NAME

    def evaluate_turn(self, turn: Turn) -> TurnEvaluation | None:
        evaluator = (
            Evaluator.objects.filter(name=self.evaluator_name)
            .order_by('-created_at')
            .first()
        )
        if not evaluator:
            return None

        config = evaluator.config_json or {}
        rating_scale = config.get('rating_scale', {'min': 1, 'max': 5})
        weights = config.get(
            'weights',
            {
                'w_progress': 0.4737,
                'w_confusion_reduction': 0.2632,
                'w_clarity': 0.1579,
                'w_engagement': 0.1053,
            },
        )

        latest_feedback = turn.feedback_entries.order_by('-created_at', '-id').first()
        if not latest_feedback:
            return None

        progress_current = self._normalize(latest_feedback.rating_perceived_progress, rating_scale)
        clarity_current = self._normalize(latest_feedback.rating_clarity_understanding, rating_scale)
        engagement_current = self._normalize(latest_feedback.rating_engagement_fit, rating_scale)

        previous_feedback = (
            TurnFeedback.objects.filter(
                turn__conversation_id=turn.conversation_id,
                user_id=latest_feedback.user_id,
                id__lt=latest_feedback.id,
            )
            .order_by('-created_at', '-id')
            .first()
        )
        if previous_feedback is None:
            progress = 0.5
            confusion_reduction = 0.5
        else:
            prev_progress = self._normalize(previous_feedback.rating_perceived_progress, rating_scale)
            prev_clarity = self._normalize(previous_feedback.rating_clarity_understanding, rating_scale)
            progress = self._delta_to_score(progress_current - prev_progress)
            confusion_reduction = self._delta_to_score(clarity_current - prev_clarity)

        q_total = (
            weights.get('w_progress', 0.0) * progress
            + weights.get('w_confusion_reduction', 0.0) * confusion_reduction
            + weights.get('w_clarity', 0.0) * clarity_current
            + weights.get('w_engagement', 0.0) * engagement_current
        )
        q_total = max(0.0, min(1.0, q_total))

        evaluation, _ = TurnEvaluation.objects.update_or_create(
            turn=turn,
            evaluator=evaluator,
            defaults={
                'q_total': q_total,
                'q_progress': progress,
                'q_confusion_reduction': confusion_reduction,
                'q_clarity': clarity_current,
                'q_engagement': engagement_current,
            },
        )
        return evaluation

    def evaluate_turn_with_reward_payload(self, turn: Turn) -> tuple[TurnEvaluation | None, EvaluationRewardPayload | None]:
        evaluation = self.evaluate_turn(turn)
        if evaluation is None:
            return None, None
        return evaluation, EvaluationRewardPayload(
            evaluator_name=evaluation.evaluator.name,
            q_total=float(evaluation.q_total),
            q_progress=float(evaluation.q_progress),
            q_confusion_reduction=float(evaluation.q_confusion_reduction),
            q_clarity=float(evaluation.q_clarity),
            q_engagement=float(evaluation.q_engagement),
        )

    def _normalize(self, rating: int, rating_scale: dict) -> float:
        min_r = float(rating_scale.get('min', 1))
        max_r = float(rating_scale.get('max', 5))
        if max_r <= min_r:
            return 0.0
        normalized = (float(rating) - min_r) / (max_r - min_r)
        return max(0.0, min(1.0, normalized))

    def _delta_to_score(self, delta: float) -> float:
        return max(0.0, min(1.0, 0.5 + (0.5 * float(delta))))
