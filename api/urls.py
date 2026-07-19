from django.urls import path
from api.views import PredictView, MetaView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("meta/", MetaView.as_view(), name="meta"),
]
