"""Common configuration settings"""

# ======= Flask App =======
DEBUG = True
TESTING = False
PORT = 9000

# ======= Download Model Path from S3 ==============
SIBLYA_BUCKET_NAME = 'incubator-data-science-773480812817-ap-southeast-1'
SIBLYA_KEYS = ['models/comment_review/comment_review_au/comment_main_dnn_dev_kfc_location_model_79.h5',
               'models/comment_review/comment_review_au/comment_au_main_dev_kfc_location_pre_model_79.pkl']

KFC_BUCKET_NAME = 'gdp-datascience'
KFC_PIPELINE_KEY = 'development/seekau-content-filter/pipeline/filter-10_2'
KFC_MODEL_KEY = 'development/seekau-content-filter/model/filter-10_2.h5'
KFC_STOPWORDS_KEY = 'development/seekau-content-filter/input/stopwords_20180309.csv'


# ======= Machine Learning Model Config ============
BASE_MODEL_PATH = '/tmp/models/'
BASE_PIPELINE_PATH = '/tmp/pipelines/'
BASE_INPUT_PATH = '/tmp/inputs/'
PRE_MODEL_FILE_MAIN = 'comment_au_main_dev_kfc_location_pre_model_79.pkl'
MODEL_FILE = 'comment_main_dnn_dev_kfc_location_model_79.h5'

KFC_PIPELINE_PATH = 'filter-10_2'
KFC_MODEL_PATH = 'filter-10_2.h5'
KFC_STOPWORDS_PATH = 'stopwords_20180309.csv'
