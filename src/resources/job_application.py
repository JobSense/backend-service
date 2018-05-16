
"""Siblya content moderation engine service core resources"""
from flask import jsonify, request
from flask_restful import Resource
from jsonschema import validate

from src.modules.content_moderator.siblya import handler


class JobApplication(Resource):
    def post(self):
        """
        Review payload schema:
            {
                "id": "1234567",
                "title": "good company",
                "pros": "the people are friendly, a lot of benefits",
                "cons": "a lot of bullying managers",
                "workLocation": "Australia",
                "kfcPred": 0
            }
        """
        result = []
        request_payload = request.get_json()

        schema = {
            "id": u'string',
            "title": u'string',
            "pros": u'string',
            "cons": u'string',
            "workLocation": u'string',
            "kfcPred": 'integer'
        }

        try:
            for review in request_payload['data']:
                validate(review, schema)
                result.append(handler(review))
        except Exception as e:
            result = e.message

        return jsonify({'data': result})
