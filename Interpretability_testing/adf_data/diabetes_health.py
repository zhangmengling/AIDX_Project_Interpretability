import sys
sys.path.append("../")

import numpy as np
import sys
import zipfile
import os

from adf_utils.config import pConfig

sys.path.append("../")

def unzip(path, sPath):
    r = zipfile.is_zipfile(path)

    if r:
        fz = zipfile.ZipFile(path, 'r')
        for file in fz.namelist():
            fz.extract(file, sPath)
    else:
        print('This is not zip')

def diabetes_health_data(*self):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    # unzip("../datasets/diabetes.csv.zip", "../datasets/diabetes_data.csv")

    unzip(os.path.join(pConfig.zip_path, "datasets", "diabetes_health.zip"), pConfig.data_path)

    with open(pConfig.data_path + "diabetes_health/diabetes.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # print(line1)
            # L = list(map(int, line1[:-1]))
            L = [int(x) if x.isdigit() else float(x) for x in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 8)
    nb_classes = 2

    return X, Y, input_shape, nb_classes
