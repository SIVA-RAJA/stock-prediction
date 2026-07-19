import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from typing import cast, Any

from api.serializers import PredictRequestSerializer
from LSTM.predict import predict

log = logging.getLogger(__name__)



class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict[str, Any], serializer.validated_data)

        try:
            result = predict(
                ticker=data["ticker"],
                market=data["market"],
                region=data["region"],
                interval=data["interval"],
            )
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as e:
            log.exception("Prediction failed")
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        except Exception as e:
            # Catch-all so a bug anywhere in the pipeline (pandas/numpy/
            # yfinance internals, etc.) still comes back as JSON instead of
            # Django's HTML error page, which was crashing the Streamlit
            # client's resp.json() call. Full traceback goes to the logs.
            log.exception("Unexpected error during prediction")
            return Response({"error": f"Unexpected error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result, status=status.HTTP_200_OK)

class MetaView(APIView):
    """Expose valid tickers/markets/regions/intervals for frontend dropdowns."""
    def get(self, request):
        from data.config import TICKERS, CONFIGS, COMPANY_NAMES
        return Response({
            "tickers": TICKERS,
            "intervals": list(CONFIGS.keys()),
            "company_names": COMPANY_NAMES,
        })
