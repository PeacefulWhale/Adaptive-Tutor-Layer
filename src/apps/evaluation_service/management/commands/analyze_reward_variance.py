from __future__ import annotations

import json
import statistics
from datetime import datetime, time

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime

from apps.prompt_service.models import PromptDecision


class Command(BaseCommand):
    help = "Compare reward variance between two reward versions."

    def add_arguments(self, parser):
        parser.add_argument('--baseline-version', default='mean_q_v0')
        parser.add_argument('--candidate-version', default='q_v2')
        parser.add_argument('--user-id', default=None)
        parser.add_argument('--since', default=None, help='ISO datetime or date')
        parser.add_argument('--fail-if-not-improved', action='store_true')

    def handle(self, *args, **options):
        baseline_version = str(options['baseline_version'])
        candidate_version = str(options['candidate_version'])
        user_id = options.get('user_id')
        since = options.get('since')
        fail_if_not_improved = bool(options.get('fail_if_not_improved'))

        qs = PromptDecision.objects.filter(reward__isnull=False)
        if user_id:
            qs = qs.filter(learner_id=user_id)
        if since:
            since_dt = self._parse_since(since)
            qs = qs.filter(chosen_at__gte=since_dt)

        baseline_scores = list(
            qs.filter(reward_version=baseline_version).values_list('reward', flat=True)
        )
        candidate_scores = list(
            qs.filter(reward_version=candidate_version).values_list('reward', flat=True)
        )

        baseline_stats = _score_stats(baseline_scores)
        candidate_stats = _score_stats(candidate_scores)
        baseline_var = baseline_stats['variance']
        candidate_var = candidate_stats['variance']
        variance_delta = None
        improved = None
        if baseline_var is not None and candidate_var is not None:
            variance_delta = candidate_var - baseline_var
            improved = variance_delta > 0

        payload = {
            'baseline_version': baseline_version,
            'candidate_version': candidate_version,
            'user_id': user_id,
            'since': since,
            'baseline': baseline_stats,
            'candidate': candidate_stats,
            'variance_delta': variance_delta,
            'improved': improved,
        }
        self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))

        if fail_if_not_improved and improved is not True:
            raise CommandError(
                "Candidate variance did not improve over baseline "
                f"({baseline_version} -> {candidate_version})."
            )

    def _parse_since(self, raw_since: str) -> datetime:
        dt = parse_datetime(raw_since)
        if dt is not None:
            if timezone.is_naive(dt):
                return timezone.make_aware(dt, timezone.get_current_timezone())
            return dt

        d = parse_date(raw_since)
        if d is not None:
            dt = datetime.combine(d, time.min)
            return timezone.make_aware(dt, timezone.get_current_timezone())

        raise CommandError(f"Invalid --since value '{raw_since}'. Use ISO date or datetime.")


def _score_stats(scores: list[float]) -> dict:
    count = len(scores)
    mean = float(sum(scores) / count) if count else None
    variance = float(statistics.pvariance(scores)) if count >= 2 else None
    return {
        'count': count,
        'mean': mean,
        'variance': variance,
    }
