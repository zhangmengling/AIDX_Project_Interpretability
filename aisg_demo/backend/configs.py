import os

class pConfig:
    aPath = os.path.abspath('.')
    path = aPath[:aPath.index('project')] + 'project/'
    zip_path = path + "sav_demo/main/static/compressed/"
    data_path = path + "sav_demo/main/static/datasets/"
    model_path = path + "sav_demo/main/static/models/"
    explanation_path = path + "sav_demo/main/static/explanation/"
