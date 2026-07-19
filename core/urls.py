from django.urls import path, include
from django.http import JsonResponse


def health_check(request):

    return JsonResponse({
        "status": "ok",
        "service": "Stock Direction Predictor API",
        "endpoints": {
            "meta": "/api/meta/",
            "predict": "/api/predict/ (POST)",
        },
    })


urlpatterns = [
    path('', health_check, name='health-check'),
    path('api/', include('api.urls')),
]
