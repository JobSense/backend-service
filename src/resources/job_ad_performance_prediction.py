
"""Siblya content moderation engine service core resources"""
from flask import jsonify, request
from flask_restful import Resource


def get_where_clause(payload):
    return ''


class JobAdPerformance(Resource):
    def post(self):
        result = dict()
        request_payload = request.get_json()
        print(request_payload)

        try:
            # cursor = current_app.mysql.connection.cursor()
            # query = """
            # select
            #     min(job_total_reach) as reach_0_percentile,
            #     avg(job_total_reach) as reach_50_percentile,
            #     max(job_total_reach) as reach_100_percentile,
            #     min(job_total_view) as view_0_percentile,
            #     avg(job_total_view) as view_50_percentile,
            #     max(job_total_view) as view_100_percentile,
            #     min(job_total_application) as application_0_percentile,
            #     avg(job_total_application) as application_50_percentile,
            #     max(job_total_application) as application_100_percentile,
            #     min()
            #     count(1) as group_size
            # from
            #     job_ad
            # where
            #     job_total_application != 0
            #     {where_clause}
            #     and job_employment_type = 'Full-Time'
            #     and job_seniority_level = 'Manager'
            #     and job_specialization_s = 'Architecture/Interior Design'
            #     and job_role_s = 'Architect'
            # group by
            #     job_employment_type,
            #     job_seniority_level,
            #     job_specialization_s,
            #     job_role_s
            # """.format(where_clause=get_where_clause(request_payload))

            # cursor.execute(query)
            # data = cursor.fetchone()
            # cursor.close()

            # print(data)
            print('')

            
        except Exception as e:
            result = e.message

        return jsonify({'data': result})
