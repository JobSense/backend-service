
"""Siblya content moderation engine service core resources"""
from flask import jsonify, request, make_response
from flask_restful import Resource
from flask import current_app
import numpy as np
from pprint import pprint

from modules.performance_predictor.package_hackathon import handler


def get_where_clause(p_inp):
    query = ''
    for k, v in p_inp.iteritems():
        if v != '':
            query += "and {k} = '{v}' ".format(k=k, v=v)

    return query

def get_group_clause(p_inp):
    base_query = 'group by '
    arr = []
    for k, v in p_inp.iteritems():
        if v != '':
            arr.append(k)
    
    group_query = ','.join(arr)
    if len(arr) > 0:
        return base_query + group_query
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
            selected_payload = {
                'job_industry': request_payload.get('job_industry', ''),
                'job_employment_type': request_payload.get('job_employment_type', ''),
                'job_seniority_level': request_payload.get('job_seniority_level', ''),
                'job_specialization_s': request_payload.get('job_specializations_string', ''), # TODO: should be array
                'job_role_s': request_payload.get('job_roles_string', ''), # TODO: should be array
                'job_work_location_s': request_payload.get('job_work_locations_string', ''), # TODO: should be array, if multi location selected, ignore this field
                'years_of_experience': request_payload.get('years_of_experience', '')
            }
            
            cursor = current_app.mysql.connection.cursor()
            query = """
            select
                min(job_total_reach) as reach_0_percentile,
                avg(job_total_reach) as reach_50_percentile,
                max(job_total_reach) as reach_100_percentile,
                min(job_total_view) as view_0_percentile,
                avg(job_total_view) as view_50_percentile,
                max(job_total_view) as view_100_percentile,
                min(job_total_application) as application_0_percentile,
                avg(job_total_application) as application_50_percentile,
                max(job_total_application) as application_100_percentile,
                min(cast(job_monthly_salary_min as decimal(10,3))) as salary_0_percentile,
                avg((cast(job_monthly_salary_max as decimal(10,3)) + cast(job_monthly_salary_min as decimal(10,3)))/2) as salary_50_percentile,
                max(job_monthly_salary_max) as salary_100_percentile,
                count(1) as group_size
            from
                job_ad
            where
                job_total_reach > 10
                and job_total_view != 0
                and job_total_application != 0
                and job_monthly_salary_min != 0
            {where_clause}
            {group_clause}
            """.format(where_clause=get_where_clause(selected_payload), group_clause=get_group_clause(selected_payload))

            cursor.execute(query)
            data = cursor.fetchone()
            cursor.close()

            print(query)

            if data:
                min_reach = float(data['reach_0_percentile'])
                max_reach = float(data['reach_100_percentile'])
                median_reach = float(data['reach_50_percentile'])

                min_view = float(data['view_0_percentile'])
                max_view = float(data['view_100_percentile'])
                median_view = float(data['view_50_percentile'])

                min_app = float(data['application_0_percentile'])
                max_app = float(data['application_100_percentile'])
                median_app = float(data['application_50_percentile'])

                min_salary = float(data['salary_0_percentile'])
                max_salary = float(data['salary_100_percentile'])
                median_salary = float(data['salary_50_percentile'])

                result['insights'] = {
                    "talentPool": {
                        'min': min_reach,
                        'max': max_reach,
                        'median': median_reach,
                        'toShow': True
                    },
                    "clicks": {
                        'min': min_view,
                        'max': max_view,
                        'median': median_view,
                        'toShow': True
                    },
                    "applies": {
                        'min': min_app,
                        'max': max_app,
                        'median': median_app,
                        'toShow': True
                    },
                    "salary": {
                        'min': min_salary,
                        'max': max_salary,
                        'median': median_salary,
                        'toShow': True
                    }
                }
            else:
                result['insights'] = {
                    "talentPool": {
                        'min': 0,
                        'max': 0,
                        'median': 0,
                        'toShow': False
                    },
                    "view": {
                        'min': 0,
                        'max': 0,
                        'median': 0,
                        'toShow': False
                    },
                    "applies": {
                        'min': 0,
                        'max': 0,
                        'median': 0,
                        'toShow': False
                    },
                    "salary": {
                        'min': 0,
                        'max': 0,
                        'median': 0,
                        'toShow': False
                    }
                }

        except Exception as e:
            result = {
                'error': e.message
            }
            print(e.message)

        return make_response(jsonify(result), 200, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS', 'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token'})
