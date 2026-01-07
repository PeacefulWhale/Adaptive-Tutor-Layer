from django.db import models


class Prompt(models.Model):
    version = models.IntegerField()
    text = models.TextField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-version', '-created_at']

    def __str__(self):
        return f"Prompt(v{self.version}, active={self.is_active})"

# Create your models here.
