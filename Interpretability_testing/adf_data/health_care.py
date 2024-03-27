import numpy as np
import sys
sys.path.append("../")
import os

from adf_utils.config import pConfig

def health_care_data(*self):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(pConfig.tmp_path + "datasets/heart_failure_clinical_records_dataset.csv", "r") as ins:
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

    input_shape = (None, 12)
    nb_classes = 2

    return X, Y, input_shape, nb_classes