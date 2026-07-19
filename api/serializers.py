from rest_framework import serializers
from data.config import TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID

class PredictRequestSerializer(serializers.Serializer):
    ticker = serializers.CharField()
    market = serializers.CharField()
    region = serializers.CharField()
    interval = serializers.CharField()

    def validate_ticker(self, v):
        if v not in TICKER_TO_ID:
            raise serializers.ValidationError(f"Unknown ticker '{v}'")
        return v

    def validate_market(self, v):
        if v not in MARKET_TO_ID:
            raise serializers.ValidationError(f"Unknown market '{v}'")
        return v

    def validate_region(self, v):
        if v not in REGION_TO_ID:
            raise serializers.ValidationError(f"Unknown region '{v}'")
        return v

    def validate_interval(self, v):
        if v not in INTERVAL_TO_ID:
            raise serializers.ValidationError(f"Unknown interval '{v}'")
        return v
