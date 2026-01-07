from django.urls import path

from apps.api.views import TutorRespondView

urlpatterns = [
    path('tutor/respond', TutorRespondView.as_view(), name='tutor-respond'),
]
