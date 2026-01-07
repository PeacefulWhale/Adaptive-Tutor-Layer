from django.contrib import admin

from apps.history_service.models import Conversation, Turn


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('conversation_id', 'user_id', 'created_at', 'updated_at')
    search_fields = ('conversation_id', 'user_id')


@admin.register(Turn)
class TurnAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'turn_index', 'created_at')
    list_filter = ('conversation',)
    search_fields = ('conversation__conversation_id', 'user_text', 'assistant_text')

# Register your models here.
