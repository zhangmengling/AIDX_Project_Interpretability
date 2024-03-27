from __future__ import division

import sys

sys.path.append("../")
import numpy as np
import random
import time
from scipy.optimize import basinhopping
import tensorflow as tf
from tensorflow.python.platform import flags
import copy

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank
from adf_tutorial.utils import cluster, gradient_graph

from six.moves import xrange


FLAGS = flags.FLAGS

class Local_Perturbation(object):

    def __init__(self, sess, preds, x, conf, sensitive_param, param_probability, param_probability_change_size, direction_probability, direction_probability_change_size, perturbation_unit, stepsize=1):
        self.sess = sess
        self.preds = preds
        self.x = x
        self.conf = conf
        self.sensitive_param = sensitive_param
        self.param_probability = param_probability
        self.param_probability_change_size = param_probability_change_size
        self.direction_probability = direction_probability
        self.direction_probability_change_size = direction_probability_change_size
        self.perturbation_unit = perturbation_unit
        self.stepsize = stepsize
        """
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """

    def __call__(self, x):
        print("-->local_perturbation")
        '''
        print("-->local_perturbation.print parameters:")
        print(self.sess, type(self.sess))
        print(self.preds, type(self.preds))
        print(self.x, type(self.x))
        print(self.conf, type(self.conf))
        print(self.sensitive_param, type(self.sensitive_param))
        print(self.param_probability, type(self.param_probability))   #local_perturbation每个param的probability
        print(self.param_probability_change_size, type(self.param_probability_change_size))
        print(self.direction_probability, type(self.direction_probability))
        print(self.direction_probability_change_size, type(self.direction_probability_change_size))
        print(self.perturbation_unit, type(self.perturbation_unit))
        print(self.stepsize, type(self.stepsize))
        '''

        s = self.stepsize
        param_choice = np.random.choice(xrange(self.conf.params) , p=self.param_probability) #从13个选出一个param
        print("-->param_choice:", param_choice)
        print("-->xchange(self.conf.params):", xrange(self.conf.params), self.conf.params)
        perturbation_options = [-1, 1]

        # choice = np.random.choice(perturbation_options)
        direction_choice = np.random.choice(perturbation_options, p=[self.direction_probability[param_choice],
                                                                     (1 - self.direction_probability[param_choice])])

        if (x[param_choice] == self.conf.input_bounds[param_choice][0]) or (x[param_choice] == self.conf.input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)
        #prevent out of bounds for each param

        x[param_choice] = x[param_choice] + (direction_choice * self.perturbation_unit)
        print("-->x[param_choice]:", x[param_choice])

        #防止param值超出范围
        x[param_choice] = max(self.conf.input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(self.conf.input_bounds[param_choice][1], x[param_choice])
        print("-->conf.input_bounds:", self.conf.input_bounds[param_choice][0], self.conf.input_bounds[param_choice][1])
        print("-->x[param_choice]:", x[param_choice])

        ei = check_for_error_condition(self.conf, self.sess, self.x, self.preds, x, self.sensitive_param)
        #ei: slef.x 和 x 不同label时sensitive_param的取值
        #两种情况：x和slef.x不同label
        #        x和slef.x相同label

        #change direction_probability[param_choice]
        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            self.direction_probability[param_choice] = min(self.direction_probability[param_choice] +
                                                      (self.direction_probability_change_size * self.perturbation_unit), 1)

        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            self.direction_probability[param_choice] = max(self.direction_probability[param_choice] -
                                                      (self.direction_probability_change_size * self.perturbation_unit), 0)

        if ei: #ei = 1
            self.param_probability[param_choice] = self.param_probability[param_choice] + self.param_probability_change_size
            self.normalise_probability()
        else: #ei = 0
            self.param_probability[param_choice] = max(self.param_probability[param_choice] - self.param_probability_change_size, 0)
            self.normalise_probability()

        return x

    def normalise_probability(self):
        probability_sum = 0.0
        for prob in self.param_probability:
            probability_sum = probability_sum + prob

        for i in range(self.conf.params):
            self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)


#生成一个随机x
class Global_Discovery(object):
    def __init__(self, conf, stepsize=1):
        self.conf = conf
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        for i in xrange(self.conf.params): #13个param
            random.seed(time.time())
            x[i] = random.randint(self.conf.input_bounds[i][0], self.conf.input_bounds[i][1])

        # x[sensitive_param - 1] = 0
        return x

def check_for_error_condition(conf, sess, x, preds, t, sens): #t is 1d array
    t = np.array(t).astype("int")    #改变数据类型为int
    label = model_argmax(sess, x, preds, np.array([t]))  #computes the current class prediction

    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != int(t[sens-1]):
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val #改变sensitive_param的值
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                # return True
                return val  #如果仅仅改变了sensitive_param的值 label就发生改变， 返回改变的sensitive_param的值
    # return False
    return t[sens - 1]

def aequitas(dataset, sensitive_param, sample_limit, perturbation_unit):
    print("-->dataset:", dataset, type(dataset))
    print("-->sensitive_param:", sensitive_param, type(sensitive_param))
    print("-->sample_limit:", sample_limit, type(sample_limit))
    print("-->perturbation_unit:", perturbation_unit, type(perturbation_unit))

    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    data_config = {"census": census, "credit": credit, "bank": bank}

    random.seed(time.time())
    start_time = time.time()

    params = data_config[dataset].params

    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = 1000
    local_iteration_limit = 1000

    X, Y, input_shape, nb_classes = data[dataset](sensitive_param)

    print("-->X:", X)
    print("-->Y:", Y)
    print("-->input_shape:", input_shape)
    print("-->nb_classes:", nb_classes)

    model = dnn(input_shape, nb_classes)
    print("-->model:",model)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    preds = model(x)

    tf.set_random_seed(1234)   #使所有op产生的随机序列在会话之间是可重复的  同一个data生成的随机数相同
    config = tf.ConfigProto()   #配置tf.Session的运算方式
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  #占用80%显存 限制gpu使用率
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, '../models/' + dataset + '/999/test.model')

    if dataset == "census":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "credit":
        initial_input = [2,24,2,2,37,0,1,2,1,0,4,2,2,2,1,1,2,1,0,0]
    elif dataset == "bank":
        initial_input = [3,11,2,0,0,5,1,0,0,5,4,40,1,1,0,0]
    minimizer = {"method": "L-BFGS-B"}

    def evaluate_local(inp):
        print("-->evaluate_local.inp:", inp)
        result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)
        # if len(tot_inputs) < limit:
        temp = copy.deepcopy(inp.astype('int').tolist())
        print("-->evaluate_local.temp:", temp)
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
        tot_inputs.add(tuple(temp))
        #存在discriminatory
        if result != int(inp[sensitive_param-1]) and (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
            local_disc_inputs.add(tuple(temp))
            local_disc_inputs_list.append(temp)

        return not result           # for binary classification, we have found that the
                                    # following optimization function gives better results
                                    # return abs(out1 + out0)

    global_discovery = Global_Discovery(data_config[dataset])
    local_perturbation = Local_Perturbation(sess, preds, x, data_config[dataset],sensitive_param, param_probability,param_probability_change_size,direction_probability,direction_probability_change_size,perturbation_unit)

    length = min(sample_limit, len(X))
    print("-->length:", length)
    value_list = []
    for i in range(length):
        print("-->i:", i)
        print("-->initial_input:", initial_input)
        inp = global_discovery.__call__(initial_input)
        print("-->after gloabal_discovery(inp):", inp)
        temp = copy.deepcopy(inp)
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:] #去掉sensitive param
        tot_inputs.add(tuple(temp))

        result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)

        # 存在discriminatory
        if result != inp[sensitive_param - 1] and (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
            global_disc_inputs_list.append(temp)
            global_disc_inputs.add(tuple(temp))
            # print(result)
            value_list.append([inp[sensitive_param - 1], result])

            basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                         niter=1000)
            print(len(global_disc_inputs), len(local_disc_inputs),
                  "Percentage discriminatory inputs of local search- " + str(
                      float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100))

            # print(i,"Percentage discriminatory inputs - " +
            #       str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
        # if len(tot_inputs) >= limit:
        #     break

    # global_disc_inputs_list = np.load('../results/census/gender/samples_1_10_1000.npy')



    print("-->directory:" + '../results/'+dataset+'/'+ str(sensitive_param) + '/global_samples_aequitas.npy')
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/global_samples_aequitas.npy', np.array(global_disc_inputs_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/disc_value_aequitas.npy', np.array(value_list))
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_aequitas.npy', np.array(local_disc_inputs_list))
    print("")
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(global_disc_inputs)+len(local_disc_inputs)))

def main(argv=None):
    aequitas(dataset=FLAGS.dataset,
             sensitive_param =FLAGS.sens_param,
             sample_limit=FLAGS.sample_limit,
             perturbation_unit=FLAGS.perturbation_unit)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sample_limit', 1000, 'number of samples to search')
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_float('perturbation_unit', 1.0, 'step size for perturbation')

    tf.app.run()





'''
    global_discovery = Global_Discovery(data_config[dataset])
    local_perturbation = Local_Perturbation(sess, preds, x, data_config[dataset],sensitive_param, param_probability,param_probability_change_size,direction_probability,direction_probability_change_size,perturbation_unit)

    # global_disc_inputs_list = np.load('../results/census/8/global_samples.npy')
    # value_list = np.load('../results/census/8/disc_value.npy')
    # print('../results/census/8/global_samples.npy')
    # for i in range(len(global_disc_inputs_list)):
    #     inp = global_disc_inputs_list[i].tolist()
    #     inp = inp[:sensitive_param-1] + [value_list[i][0]] + inp[sensitive_param-1:]
    #     basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
    #                          niter=1000)
    #     print(i, len(local_disc_inputs),
    #                   "Percentage discriminatory inputs of local search- " + str(
    #                       float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100))
'''