from rest_framework import serializers


class TutorRespondRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    conversation_id = serializers.CharField(required=False, allow_blank=True)
    question_text = serializers.CharField()
