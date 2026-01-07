import json
import urllib.error
import urllib.request

from django.conf import settings

from common.errors import LLMUpstreamError


class LLMService:
    def generate(
        self,
        system_prompt_text: str,
        history_turns: list,
        user_message: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        messages = [{'role': 'system', 'content': system_prompt_text}]
        for turn in history_turns:
            messages.append({'role': 'user', 'content': turn.user_text})
            messages.append({'role': 'assistant', 'content': turn.assistant_text})
        messages.append({'role': 'user', 'content': user_message})

        payload = {
            'model': model or settings.LLM_DEFAULT_MODEL,
            'messages': messages,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'adaptive-tutor-layer/0.1',
        }
        if settings.LLM_API_KEY:
            headers['Authorization'] = f"Bearer {settings.LLM_API_KEY}"

        request = urllib.request.Request(
            settings.LLM_API_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST',
        )

        try:
            with urllib.request.urlopen(request, timeout=settings.LLM_TIMEOUT_SECONDS) as resp:
                raw_body = resp.read().decode('utf-8')
                data = json.loads(raw_body) if raw_body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8') if exc.fp else ''
            raise LLMUpstreamError(exc.code, f"Upstream error {exc.code}", body=body) from exc
        except Exception as exc:
            raise LLMUpstreamError(None, str(exc)) from exc

        assistant_text = ''
        try:
            assistant_text = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            assistant_text = data.get('output_text', '')

        return {
            'assistant_text': assistant_text,
            'raw_response': data,
            'metadata': {
                'model': payload.get('model'),
                'temperature': payload.get('temperature'),
                'max_tokens': payload.get('max_tokens'),
                'x_request_id': data.get('id'),
            },
        }
