from rest_framework import serializers

class AddDataSerializer(serializers.Serializer):
    context = serializers.CharField()
    response = serializers.CharField()
