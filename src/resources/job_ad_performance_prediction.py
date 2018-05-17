
"""Siblya content moderation engine service core resources"""
from flask import jsonify, request
from flask_restful import Resource
import numpy as np
from pprint import pprint

from modules.performance_predictor.package_hackathon import handler

def get_where_clause(payload):
    return ''

def get_num_features(payload):
    output = [
        int(payload['job_auto_forwarded_flag'] == True),
        int(payload['job_internship_flag'] == True),
        int(payload['job_salary_visible'] == True),
        int(payload['company_recruitment_firm_flag'] == True),
        payload.get('job_monthly_salary_min', 0),
        payload.get('job_monthly_salary_max', 0),
        payload.get('job_posting_date_start_datediff', 548),
        payload.get('job_posting_date_end_datediff', 578)
    ]
    return np.array(output).reshape(1,8)

class JobAdPerformance(Resource):
    def post(self):
        result = dict()
        request_payload = request.get_json()

        try:
            processed_payload = {
                'job_title': request_payload.get('job_title', ' '),
                'job_seniority_level': request_payload.get('job_seniority_level', ' '),
                'job_industry': request_payload.get('job_industry', ' '),
                'job_description': request_payload.get('job_description', ' '),
                'job_requirement': request_payload.get('job_requirement', ' '),
                'job_employment_type': request_payload.get('job_employment_type', ' '),
                'company_name': request_payload.get('company_name', ' '),
                'company_size': request_payload.get('company_size', ' '),
                'job_specializations': request_payload.get('job_specializations_string', ' '),
                'job_roles': request_payload.get('job_roles_string', ' '),
                'job_work_locations': request_payload.get('job_work_locations', ' '),
                'company_location': request_payload.get('company_location_string', ' '),
                'qualification_code': request_payload.get('qualification_code_string', ' '),
                'field_of_study': request_payload.get('field_of_study_string', ' '),
                'mandatory_skill_keyword': request_payload.get('mandatory_skill_keyword', ' '),
                'num_features': get_num_features(request_payload)
            }

            output = handler(processed_payload)
            reach = output['reach']
            view = output['view']
            application = output['application']

            result = {
                "talentPool": {
                    "prediction": {
                        "min": float(reach[1][0][0][0]),
                        "max": float(reach[1][1][0][0]),
                        "median": float(reach[0][0][0]),
                        "toShow": True
                    }
                },
                "clicks": {
                    "prediction": {
                        "min": float(view[1][0][0][0]),
                        "max": float(view[1][1][0][0]),
                        "median": float(view[0][0][0]),
                        "toShow": True
                    }
                },
                "applies": {
                    "prediction": {
                        "min": float(application[1][0][0][0]),
                        "max": float(application[1][1][0][0]),
                        "median": float(application[0][0][0]),
                        "toShow": True
                    }
                }
            }

            # Insights geh
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
            min_app = 0
            max_app = 0
            median_app = 0

            min_salary = 0
            max_salary = 0
            median_salary = 0

            result['insights'] = {
                "applies": "Based on job ad posted in the market that have similar attributes (industry, location and years of experience), the minimum application is {min_app}, maximum is {max_app} and median is {median_app}.".format(min_app=min_app, max_app=max_app, median_app=median_app),
                "salary": "Based on job ad posted in the market that have similar attributes (industry, location and years of experience), the minimum salary is {min_salary}, maxiumum is {max_salary} and median is {median_salary}.".format(min_salary=min_salary, max_salary=max_salary, median_salary=median_salary)
            }

        except Exception as e:
            result = e.message

        return jsonify({'data': result})
