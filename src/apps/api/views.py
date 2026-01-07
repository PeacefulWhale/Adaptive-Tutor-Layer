from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.handler.service import TutorResponseHandler
from common.errors import (
    LLMUpstreamError,
    PersistenceError,
    PromptDataError,
    PromptNotFoundError,
)
from .serializers import TutorRespondRequestSerializer


class TutorRespondView(APIView):
    def post(self, request):
        serializer = TutorRespondRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        conversation_id = data.get('conversation_id') or None
        model = data.get('model') or None
        handler = TutorResponseHandler()

        try:
            result = handler.generate_response(
                user_id=data['user_id'],
                conversation_id=conversation_id,
                question_text=data['question_text'],
                model=model,
                temperature=data.get('temperature'),
                max_tokens=data.get('max_tokens'),
            )
        except PromptNotFoundError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except PromptDataError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except LLMUpstreamError as exc:
            return Response(
                {'detail': str(exc), 'status': exc.status, 'body': exc.body},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except PersistenceError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {
                'conversation_id': result['conversation_id'],
                'raw_model_response': result['raw_model_response'],
                'tutor_response': result['tutor_response'],
                'prompt_version_id': result['prompt_version_id'],
            },
            status=status.HTTP_200_OK,
        )
