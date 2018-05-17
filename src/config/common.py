"""Common configuration settings"""

# ======= Flask App =======
DEBUG = True
TESTING = False
PORT = 9000

# ======= Download Model Path from S3 ==============
BUCKET_NAME = 'incubation-data-science-prod'
KEYS = [
    'models/hackathon/hackathon_main_pre_model_1.pkl',
    'models/hackathon/hackathon_main_model_1.h5'
]

# ======= Machine Learning Model Config ============
BASE_MODEL_PATH = '/tmp/models/'
BASE_PIPELINE_PATH = '/tmp/pipelines/'
BASE_INPUT_PATH = '/tmp/inputs/'
PRE_MODEL_FILE_MAIN = 'hackathon_main_pre_model_1.pkl'
MODEL_FILE = 'hackathon_main_model_1.h5'
