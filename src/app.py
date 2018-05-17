from flask import Flask, jsonify
from flask_restful import Api
from flask_mysqldb import MySQL

from src.config import common
from src.resources.job_ad_performance_prediction import JobAdPerformance
from src.modules.performance_predictor.package_hackathon import spremodel_load, smodel_load

# Create the Flask app
app = Flask(__name__)

with app.app_context():
    # Load configuration vals common
    app.config.from_object(common)

    # Load configuration vals based on env
    app.config.from_envvar('APP_ENV_CONFIG', silent=True)

    # Setup mysql
    app.config['MYSQL_USER'] = 'jobsense'
    app.config['MYSQL_PASSWORD'] = 'jobsense'
    app.config['MYSQL_DB'] = 'insights'
    app.config['MYSQL_HOST'] = 'jobsensedb.chhnlo3xihgd.ap-southeast-1.rds.amazonaws.com'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
    app.mysql = MySQL(app)

    # Initialize heavy loading content, esp. the machine learning model
    # load model
    app.pre_model = spremodel_load(
        common.BASE_MODEL_PATH + common.PRE_MODEL_FILE_MAIN)
    app.model = smodel_load(common.BASE_MODEL_PATH + common.MODEL_FILE)

print("All model loading done. Your API service are now ready~")


# Create restful api flask app, add all resources require
api = Api(app)
api.prefix = '/predictions'
api.add_resource(JobAdPerformance, '/job-ad-performance')


@app.route('/healthcheck')
def healthcheck():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'])
