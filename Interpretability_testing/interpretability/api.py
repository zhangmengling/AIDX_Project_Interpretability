import sys
sys.path.append("../")

from interpretability_testing import interpretability_testing

import os
aPath = os.path.abspath('.')
path = aPath[:aPath.index('project')] + 'project/'
sys.path.append(path + "Interpretability_testing/")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank, health_care, diabetes_health
from adf_tutorial.utils import cluster, gradient_graph

from adf_utils.config import pConfig

import argparse
import sqlite3
from datetime import datetime
import zipfile

def unzip(path, sPath):
    r = zipfile.is_zipfile(path)

    if r:
        fz = zipfile.ZipFile(path, 'r')
        for file in fz.namelist():
            fz.extract(file, sPath)
    else:
        print('This is not zip')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--caseId', type=str)
parser.add_argument('-m', '--modelFile', type=str)
parser.add_argument('-d', '--dataFile', type=str)
parser.add_argument('-l', '--layers', type=str)

args = parser.parse_args()

DB_PATH = '../../sav_demo/main/static/aidx.db'

dataPath = args.dataFile #"mnist.zip"
modelPath = args.modelFile #"mnist_lenet5.zip"

dataName = dataPath.split("/")[-1][:-4]
modelName = modelPath.split("/")[-1].split("_")[-1][:-4]

unzip(os.path.join(pConfig.zip_path, "models", modelPath), pConfig.model_path)

data2filename = {"census": "census", "credit": "credit", "bank": "bank",
                 "health_care": "heart_failure_clinical_records_dataset.csv",
                 "diabetes_health": "diabetes.csv"}
data_config = {"census": census, "credit": credit, "bank": bank, "health_care": health_care,
               "diabetes_health": diabetes_health}

datafilename = pConfig.data_path + dataName + "/" + data2filename[dataName]

output_pngfile = pConfig.explanation_path + args.caseId + ".png"
score, output_pngfile = interpretability_testing(dataName, datafilename, pConfig.model_path + modelPath.split("/")[-1][:-4] + "/" + modelName + ".model", int(args.layers), output_pngfile)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
query = "UPDATE cases SET end_time = ?, score = ? WHERE case_id = ?"
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
cursor.execute(query, (current_time, score, args.caseId))
conn.commit()

conn.close()

print("Finished")


