
"""Siblya content moderation engine service core resources"""
from flask import jsonify, request
from flask_restful import Resource
from flask import current_app

import json
import decimal

class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)

def get_category(keys):
    return ''


def get_where_clause(values):
    return ''


class JobAdPerformance(Resource):
    def get(self):
        args = request.args
        print args
        result = dict()
        try:
            cursor = current_app.mysql.connection.cursor()
            query = """
            select
                {categories}
                job_employment_type,
                job_seniority_level,
                job_specialization_s,
                job_role_s,
                min(job_total_reach) as reach_0_percentile,
                avg(job_total_reach) as reach_50_percentile,
                max(job_total_reach) as reach_100_percentile,
                min(job_total_view) as view_0_percentile,
                avg(job_total_view) as view_50_percentile,
                max(job_total_view) as view_100_percentile,
                min(job_total_application) as application_0_percentile,
                avg(job_total_application) as application_50_percentile,
                max(job_total_application) as application_100_percentile,
                count(1) as group_size
            from
                job_ad
            where
                job_total_application != 0
                {where_clause}
                and job_employment_type = 'Full-Time'
                and job_seniority_level = 'Manager'
                and job_specialization_s = 'Architecture/Interior Design'
                and job_role_s = 'Architect'
            group by
                job_employment_type,
                job_seniority_level,
                job_specialization_s,
                job_role_s
            """.format(categories=get_category(''), where_clause=get_where_clause(''))

            cursor.execute(query)

            data = cursor.fetchone()
            cursor.close()
            print(data)
            result = ''

        except Exception as e:
            result = e.message

        return jsonify(result)
