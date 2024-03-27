import os

# path configuration
class pConfig:
    aPath = os.path.abspath('.')
    path = aPath[:aPath.index('project')] + 'project/'
    zip_path = path + "sav_demo/main/static/compressed/"
    data_path = path + "sav_demo/main/static/datasets/"
    model_path = path + "sav_demo/main/static/models/"
    adv_path = path + "sav_demo/main/static/advs/"
    noise_path = path + "sav_demo/main/static/noises/"
    explanation_path = path + "sav_demo/main/static/explanation/"

# data configuration
class dConfig:
    dName = "mnist"
    iShape = (None, 28, 28, 1)
    cClasses = 10

# model configuration
class mConfig:
    mName = "lenet5"
    activation = "relu"

# running configuration
class rConfig:
    pBatch = 256 # prediction batch
    aBatch = 128 # attack batch
    tBatch = 256 # testing batch
