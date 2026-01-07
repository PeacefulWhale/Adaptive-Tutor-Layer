from django.db import models


class Conversation(models.Model):
    conversation_id = models.CharField(max_length=64, unique=True)
    user_id = models.CharField(max_length=128, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Conversation({self.conversation_id})"


class Turn(models.Model):
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='turns',
    )
    turn_index = models.IntegerField()
    user_text = models.TextField()
    assistant_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    metadata_json = models.JSONField(default=dict)

    class Meta:
        unique_together = [('conversation', 'turn_index')]
        ordering = ['turn_index']

    def __str__(self):
        return f"Turn({self.conversation.conversation_id}:{self.turn_index})"

# Create your models here.
