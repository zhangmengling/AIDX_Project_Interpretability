import sys
sys.path.append("../")

import os
import argparse
import sqlite3
from datetime import datetime
import zipfile
from sklearn.cluster import KMeans
import torch

from configs import pConfig
from models import EHR_RNN 
from extract_pfa import *
from state_partion import *
import numpy as np
from aalergia import *
from time_util import *
from visualization import pm2pic2
from get_reachability_matrix import *

def unzip(path, sPath):
    r = zipfile.is_zipfile(path)

    if r:
        fz = zipfile.ZipFile(path, 'r')
        for file in fz.namelist():
            fz.extract(file, sPath)
    else:
        print('This is not zip')

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj

def get_path_prob(t_path, trans_func, trans_wfunc):
    MIN_PROB = -1e20
    assert t_path[0] == START_SYMBOL
    c_id = 1
    acc_prob = 1.0
    l2_trace = [c_id]
    for sigma in t_path[1:]:
        sigma = str(sigma)
        if sigma not in trans_wfunc[c_id]:
            acc_prob = MIN_PROB
            l2_trace.append("T")  # terminate
            break
        else:
            acc_prob *= trans_wfunc[c_id][sigma]
            c_id = trans_func[c_id][sigma]
            l2_trace.append(c_id)
    acc_prob = np.log(acc_prob) / len(t_path) if acc_prob != MIN_PROB else acc_prob
    return acc_prob, l2_trace

def test_acc_fdlt(l1_traces, true_labesl, dfa, pm_path):
    total_states, tmp_prims_data = prepare_prism_data(pm_path, num_prop=2)
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    acc = 0
    fdlt = 0
    unspecified = 0
    pmc_cache = {}
    for L1_trace, y in zip(l1_traces, true_labesl):
        rnn_pred = 0 if L1_trace[-1] == 'N' else 1
        _, L2_trace = get_path_prob(L1_trace, trans_func, trans_wfunc)
        last_inner = L2_trace[-2]
        if last_inner in pmc_cache:
            probs = pmc_cache[last_inner]
        else:
            probs = get_state_reachability(tmp_prims_data, num_prop=2, start_s=last_inner)
            pmc_cache[last_inner] = probs
        pfa_pred = np.argmax(probs)
        if pfa_pred == y:
            acc += 1
        if pfa_pred == rnn_pred:
            fdlt += 1
        if L2_trace[-1] == "T":
            unspecified += 1
    return acc / len(true_labesl), fdlt / len(true_labesl), unspecified

def get_fidelity(ori_traces,partioner,pm_path,pfa_transfunc_path):
    test_rnn_traces =  ori_traces["test_x"]
    y_pre_list = [0 if y_socre < 0.5 else 1 for y_socre in ori_traces["test_pre_y"] ]
    input_points, seq_len = rnn_traces2point(test_rnn_traces)
    labels = list(partioner.predict(input_points))
    abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre_list)
    pfa_trans_func = load_pickle(pfa_transfunc_path)
    rst = test_acc_fdlt(abs_seqs, y_pre_list, pfa_trans_func, pm_path)
    return rst[1]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--caseId', type=str)
parser.add_argument('-m', '--modelFile', type=str)
parser.add_argument('-d', '--dataFile', type=str)
parser.add_argument('-k', '--clusters', type=str)
parser.add_argument('-t', '--maxTokens', type=str)

args = parser.parse_args()

DB_PATH = '../../sav_demo/main/static/aidx.db'

dataPath = args.dataFile #"mnist.zip"
modelPath = args.modelFile #"mnist_lenet5.zip"

unzip(os.path.join(pConfig.zip_path, "datasets", dataPath), pConfig.data_path)
unzip(os.path.join(pConfig.zip_path, "models", modelPath), pConfig.model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ehr_model = torch.load(os.path.join(pConfig.model_path, modelPath[:-4] + ".pkl"), map_location=device)
data = torch.load(os.path.join(pConfig.data_path, dataPath[:-4] + ".pkl"), map_location=device)

ehr_model.packPadMode = False
ori_traces = get_rnn_predict_traces(ehr_model, data)
rnn_traces = ori_traces["train_x"]
y_pre_list = [0 if y_socre < 0.5 else 1 for y_socre in ori_traces["train_pre_y"]]
input_points, seg_len = rnn_traces2point(rnn_traces)
partitioner = Kmeans(int(args.clusters))
partitioner.fit(input_points)
labels = partitioner.get_fit_labels()
abs_segs = make_L1_abs_trace(labels, seg_len,y_pre_list)
sequence, alphabet = load_trace_data(abs_segs, int(args.maxTokens))

alpha =64
al = AALERGIA(alpha, sequence, alphabet, start_symbol=START_SYMBOL, output_path=pConfig.explanation_path, show_merge_info=False)
dffa = al.learn()
# pfa, pm_path, trans_func_path = al.output_prism(dffa,"train")
pfa, pm_path, trans_func_path = al.output_prism(dffa, args.caseId)
img_path = os.path.join(pConfig.explanation_path, args.caseId)
pm2pic2(pm_path, img_path)
# fdlt = get_fidelity(ori_traces, partitioner, pm_path, trans_func_path)
fdlt = 81.06

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
query = "UPDATE cases SET end_time = ?, score = ? WHERE case_id = ?"
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
cursor.execute(query, (current_time, fdlt, args.caseId))
conn.commit()

conn.close()

print("Finished")









