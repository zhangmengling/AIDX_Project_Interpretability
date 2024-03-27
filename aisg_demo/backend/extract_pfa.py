import numpy as np

START_SYMBOL = 'S'
STANDARD_PATH = "STANDARD"

def rnn_traces2point(rnn_traces):
    seq_len = []
    input_points = []
    for seq in rnn_traces:
        seq_len.append(len(seq))
        for hn_state in seq:
            input_points.append(hn_state)
    input_points = np.array(input_points)
    return input_points, seq_len

def get_term_symbol(y_pre):
    if y_pre == 0:
        return 'N'
    elif y_pre == 1:
        return "P"
    else:
        raise Exception("unknown label:{}".format(y_pre))

def make_L1_abs_trace(labels, seq_len, y_pre):
    start_p = 0
    abs_seqs = []
    for size, y in zip(seq_len, y_pre):
        abs_trace = labels[start_p:start_p + size]
        term_symbol = get_term_symbol(y)
        abs_trace = [START_SYMBOL] + [str(e) for e in abs_trace] + [term_symbol]
        abs_seqs.append(abs_trace)
        start_p += size
    return abs_seqs


def load_trace_data(l1_traces, symbols_count, start_symbol=None):
    '''
    The data file should comply the following rules:
    1. each line is a sequence
    2. within a sequence, each element should be split by a comma
    DataParameters.
    ----------------------------------
    data_path:
    symbols_count: int. the total number of symbols to select.
    start_symbol: None if not specify a start symbol
    Return:
        seq_list. list. the sequence list
        alphabet. set. alphabet of selected sequences
    '''
    seq_list = []
    alphabet = set()
    cnt = 0
    for seq in l1_traces:
        remain_len = symbols_count - cnt
        if remain_len >= len(seq):
            cnt += len(seq)
        else:
            seq = seq[:remain_len]
            cnt += remain_len
        seq = [start_symbol] + seq if start_symbol is not None else seq
        seq_list.append(seq)
        alphabet = alphabet.union(set(seq))
        if symbols_count != -1 and cnt >= symbols_count:
            break
    if cnt < symbols_count:
        print("no enough data, actual load {} symbols".format(cnt))
    return seq_list, alphabet

def get_rnn_predict_traces(ehr_model, ori_data):
    ehr_model.packPadMode = False
    ori_traces = {}
    ori_traces["train_x"] = []
    ori_traces["test_x"] = []
    ori_traces["train_pre_y"] = []
    ori_traces["test_pre_y"] = []
    for sample in ori_data["train"]:
        sequence, label, seq_l, mtd = sample
        hn_trace, label_trace = ehr_model.get_predict_trace(sequence, seq_l, mtd)
        ori_traces["train_x"].append(hn_trace)
        ori_traces["train_pre_y"].append(label_trace[-1])

    for sample in ori_data["test"]:
        sequence, label, seq_l, mtd = sample
        hn_trace, label_trace = ehr_model.get_predict_trace(sequence, seq_l, mtd)
        ori_traces["test_x"].append(hn_trace)
        ori_traces["test_pre_y"].append(label_trace[-1])
    return ori_traces


