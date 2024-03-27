
import torch
from flask import Flask, request,jsonify,send_file
from flask_socketio import SocketIO, emit
import os
from flask_cors import CORS
from models import EHR_RNN 
from extract_pfa import *
from state_partion import *
import numpy as np
from aalergia import *
from time_util import *
from visualization import pm2pic2
from get_reachability_matrix import *


app = Flask(__name__)
# socketio = SocketIO(app)

WORK_SPACE = "../workspace"

@app.route('/test', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    file.save(os.path.join(WORK_SPACE,file.filename))
    
    return 'File uploaded successfully'

@app.route('/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        # 拼接文件路径
        file_path = os.path.join(WORK_SPACE, filename)
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 删除文件
            os.remove(file_path)
            return jsonify({'message': 'File deleted successfully'})
        else:
            return jsonify({'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/extract/<k>/<max_tokens>', methods=['GET'])
def extract_pfa(k,max_tokens):
    """_summary_
    1. load model
    2. load_data
    3. do_l1_abstract
    4. do_l2_abstract
    5. return pfa img
    Args:
        k (_type_): _description_
        max_tokens (_type_): _description_
    """
    ehr_model = torch.load(os.path.join(WORK_SPACE,"aisg_demo_ehr_model.pkl"))
    data =  torch.load(os.path.join(WORK_SPACE,"aisg_demo_ehr_data4pfa.pkl"))
    ehr_model.packPadMode = False
    ori_traces = get_rnn_predict_traces(ehr_model,data)
    rnn_traces =  ori_traces["train_x"]
    y_pre_list = [0 if y_socre < 0.5 else 1 for y_socre in ori_traces["train_pre_y"] ]
    input_points, seq_len = rnn_traces2point(rnn_traces)
    partitioner = Kmeans(int(k))
    partitioner.fit(input_points)
    labels = partitioner.get_fit_labels()
    abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre_list)
    sequence, alphabet = load_trace_data(abs_seqs, int(max_tokens))
    print("{}, init".format(current_timestamp()))
    alpha = 64
    al = AALERGIA(alpha, sequence, alphabet, start_symbol=START_SYMBOL, output_path=WORK_SPACE,
                    show_merge_info=False)
    print("{}, learing....".format(current_timestamp()))
    dffa = al.learn()
    print("{}, done.".format(current_timestamp()))
    pfa, pm_path,trans_func_path = al.output_prism(dffa, "train")
    img_path = os.path.join(WORK_SPACE,"pfa")
    pm2pic2(pm_path,img_path)
    fdlt = get_fidelity(ori_traces,partitioner,pm_path,trans_func_path);
    print(fdlt)
    return send_file(img_path+".png", mimetype='image/png')

def get_fidelity(ori_traces,partioner,pm_path,pfa_transfunc_path):
    test_rnn_traces =  ori_traces["test_x"]
    y_pre_list = [0 if y_socre < 0.5 else 1 for y_socre in ori_traces["test_pre_y"] ]
    input_points, seq_len = rnn_traces2point(test_rnn_traces)
    labels = list(partioner.predict(input_points))
    abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre_list)
    pfa_trans_func = load_pickle(pfa_transfunc_path)
    rst = test_acc_fdlt(abs_seqs, y_pre_list, pfa_trans_func, pm_path)
    return rst[1]
    
    
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

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj


    
    
    
    
if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(debug=False,host='0.0.0.0', port=5000)
    # socketio.run(app,host='0.0.0.0', port=5000)
    # extract_pfa(3,1000000)