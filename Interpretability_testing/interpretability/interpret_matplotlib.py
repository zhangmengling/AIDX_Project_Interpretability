import matplotlib.pyplot as plt
import itertools
from numpy import *
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
k = 3
all_feature_set = list(range(1, 13 + 1))
feature_sets = list(itertools.combinations(all_feature_set, k))
print("-->feature_sets:", feature_sets)
print("-->length:", len(feature_sets))

'''
filename = '../results/gini6'
X, Y = [], []
with open(filename, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    min_gini = 0.5
    max_gini = 0.0
    min_feature_set = feature_sets[0]
    max_feature_set = feature_sets[0]
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 4
        X.append(value[0])  # 5
        Y.append(i)
        i += 1
        if value[0] < min_gini:
            min_gini = value[0]
            min_feature_set = feature_sets[i]
        if value[0] > max_gini:
            max_gini = value[0]
            max_feature_set = feature_sets[i]


def get_figure(X, Y, if_sort, title, xlabel, ylabel, savefile):
    def sorted_figure(dict):
        sorted_dict = sorted(dictionary.items(), key=lambda d: d[1])
        # sorted_dict = sorted(dictionary.iteritems(), key=lambda val: val[1], reverse=True)
        print("sorted_dict:", sorted_dict)
        print("length of dictionary", len(sorted_dict))
        sorted_items = []
        for dicts in sorted_dict:
            item = dicts[1]
            sorted_items.append(float(item))
        print("sorted:", sorted_items)
        print(len(sorted_items))
        plt.title(title)
        plt.plot(Y, sorted_items)
        # plt.scatter(Y, sorted_items)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(savefile)
        plt.show()

    if if_sort == True:
        dictionary = dict(zip(Y, X))
        print("dictionary:", dictionary)
        length = len(dictionary)
        print("-->length", length, len(Y), len(X))
        Y = list(range(0, length))
        sorted_figure(dictionary)
    else:
        plt.scatter(Y, X)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(savefile)
        plt.show()

# get_figure(X, Y, True, "average gini index (sorted)", "feature_set", "gini index", "../figures/census/k=6/average_gini(sorted).eps")


accuracy_file = '../results/accuracy6'
X, Y = [], []
x1, x2 = [], []
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    max_accuracy = 0.0
    min_accuracy = 100.0
    max_accuracy_0 = 0.0
    min_accuracy_0 = 100.0
    max_accuracy_1 = 0.0
    min_accuracy_1 = 100.0
    min_feature_set = feature_sets[0]
    for line in lines:  # 3
        value = [float(s) for s in line.split(" ")]  # 4
        X.append(value[0]/100.0)
        x1.append(value[1]/100.0)
        x2.append(value[2]/100.0)
        Y.append(i)
        i += 1
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            min_feature_set = feature_sets[i]
        if value[0] < min_accuracy:
            min_accuracy = value[0]
            max_feature_set = feature_sets[i]
        if value[1] > max_accuracy_0:
            max_accuracy_0 = value[1]
            min_feature_set_0 = feature_sets[i]
        if value[1] < min_accuracy_0:
            min_accuracy_0 = value[1]
            max_feature_set_0 = feature_sets[i]
        if value[2] > max_accuracy_1:
            max_accuracy_1 = value[2]
            min_feature_set_1 = feature_sets[i]
        if value[2] < min_accuracy_1:
            min_accuracy_1 = value[2]
            max_feature_set_1 = feature_sets[i]

# get_figure(X, Y, True, "accuracy (sorted)", "featrue_set", "accuracy(%)", "../figures/census/k=6/average_accuracy(sorted).eps")
'''




def matplot(X, Y, if_sort, title, xlabel, ylabel):
    def sorted_figure(dict):
        sorted_dict = sorted(dictionary.items(), key=lambda d: d[1])
        # sorted_dict = sorted(dictionary.iteritems(), key=lambda val: val[1], reverse=True)
        print("sorted_dict:", sorted_dict)
        print("length of dictionary", len(sorted_dict))
        sorted_items = []
        for dicts in sorted_dict:
            item = dicts[1]
            sorted_items.append(float(item))
        print("sorted:", sorted_items)
        print(len(sorted_items))
        plt.title(title)
        plt.plot(Y, sorted_items)
        # plt.scatter(Y, sorted_items)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    if if_sort == True:
        dictionary = dict(zip(Y, X))
        print("dictionary:", dictionary)
        length = len(dictionary)
        print("-->length", length, len(Y), len(X))
        Y = list(range(0, length))
        sorted_figure(dictionary)
    else:
        plt.scatter(Y, X)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


def matplot_three(X1, Y, X2, X3, title, x1label, x2label, x3label, xlabel, ylabel):
    dictionary = dict(zip(Y, X1))
    print(dictionary)
    print(len(dictionary))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    print("second sort", sorted_dict)
    print(len(sorted_dict))

    sorted_feature = []
    sorted_x2 = []
    sorted_x3 = []
    for dicts in sorted_dict:
        sorted_feature.append(dicts[0])
        sorted_x2.append(X2[dicts[0]])
        sorted_x3.append(X3[dicts[0]])
    sorted_items = []
    for dicts in sorted_dict:
        item = dicts[1]
        sorted_items.append(float(item))
    plt.title(title)
    plt.plot(Y, sorted_items, label=x1label)
    plt.plot(Y, sorted_x2, label=x2label)
    plt.plot(Y, sorted_x3, label=x3label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def matplot_three(X1, Y, X2, X3, title, x1label, x2label, x3label, xlabel, ylabel):
    dictionary = dict(zip(Y, X1))
    print(dictionary)
    print(len(dictionary))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    print("second sort", sorted_dict)
    print(len(sorted_dict))

    sorted_feature = []
    sorted_x2 = []
    sorted_x3 = []
    for dicts in sorted_dict:
        sorted_feature.append(dicts[0])
        sorted_x2.append(X2[dicts[0]])
        sorted_x3.append(X3[dicts[0]])
    sorted_items = []
    for dicts in sorted_dict:
        item = dicts[1]
        sorted_items.append(float(item))
    plt.title(title)
    plt.plot(Y, sorted_items, label=x1label)
    plt.plot(Y, sorted_x2, label=x2label)
    plt.plot(Y, sorted_x3, label=x3label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def matplot_four(X1, Y, X2, X3, X4, title, x1label, x2label, x3label, x4label, xlabel, ylabel):
    dictionary = dict(zip(Y, X1))
    print(dictionary)
    print(len(dictionary))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    print("second sort", sorted_dict)
    print(len(sorted_dict))

    sorted_feature = []
    sorted_x2 = []
    sorted_x3 = []
    sorted_x4 = []
    for dicts in sorted_dict:
        sorted_feature.append(dicts[0])
        sorted_x2.append(X2[dicts[0]])
        sorted_x3.append(X3[dicts[0]])
        sorted_x4.append(X4[dicts[0]])
    sorted_items = []
    for dicts in sorted_dict:
        item = dicts[1]
        sorted_items.append(float(item))
    plt.title(title)
    plt.plot(Y, sorted_items, label=x1label)
    plt.plot(Y, sorted_x2, label=x2label)
    plt.plot(Y, sorted_x3, label=x3label)
    plt.plot(Y, sorted_x4, label=x4label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def matplot_two(X1, X2, title, x1label, x2label, xlabel, ylabel):

    if len(X1) != len(X2):
        length = min(len(X1), len(X2))
        Y = list(range(0, length))
    else:
        Y = list(range(1, len(X1) + 1))
    print("-->lengths", len(X1), len(X2), len(Y))
    dictionary = dict(zip(Y, X1))
    print(dictionary)
    print(len(dictionary))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    print("second sort", sorted_dict)
    print(len(sorted_dict))


    sorted_feature = []
    sorted_x2 = []
    for dicts in sorted_dict:
        sorted_feature.append(dicts[0])
        sorted_x2.append(X2[dicts[0] - 1])
    sorted_items = []
    for dicts in sorted_dict:
        item = dicts[1]
        sorted_items.append(float(item))
    # plt.title(title)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(Y, sorted_items, 'darkcyan', label=x1label)
    ax1.set_ylabel(x1label)
    ax2 = ax1.twinx()
    # ax2 = fig.add_subplot(111)
    l2 = ax2.plot(Y, sorted_x2, 'darkorange', label=x2label)
    # l2 = ax2.plot(Y, sorted_x2, c = 'darkorange', label=x2label)
    ax2.set_ylabel(x2label)
    plt.xlabel(xlabel)

    ax1.set_title(title)
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    # file_name = save_file + title + ".eps"
    # plt.savefig(file_name)
    # plt.tight_layout()
    # plt.show()
    plt.show()

def matplot_two1(X1, X2, title, x1label, x2label, xlabel, ylabel):

    if len(X1) != len(X2):
        length = min(len(X1), len(X2))
        Y = list(range(0, length))
    else:
        Y = list(range(1, len(X1) + 1))
    print("-->lengths", len(X1), len(X2), len(Y))
    dictionary = dict(zip(Y, X1))
    print(dictionary)
    print(len(dictionary))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    print("second sort", sorted_dict)
    print(len(sorted_dict))

    sorted_feature = []
    sorted_x2 = []
    for dicts in sorted_dict:
        sorted_feature.append(dicts[0])
        sorted_x2.append(X2[dicts[0] - 1])
    sorted_items = []
    for dicts in sorted_dict:
        item = dicts[1]
        sorted_items.append(float(item))
    plt.title(title)
    plt.plot(Y, sorted_items, label=x1label)
    plt.plot(Y, sorted_x2, label=x2label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# savefile = "../perturbation_figures/census/k=4/"

ci_num_file = "../result/census/k=4/CI_num"
Y = []
ci = []
seed = []
ci_seed = []
with open(ci_num_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 4
        ci.append(value[0])  # 5
        seed.append(value[1])
        ci_seed.append(value[2])
        Y.append(i)
        i += 1

# matplot_three(ci_seed, Y, ci, seed, "counter instance & seed instance(based on proportion)", "CI/seed", "CI", "seed instance", "feature_set", "#instance")


iteration_file = "../result/census/k=4/iteration_num"
ite_num = []
ite_proportion1 = []
with open(iteration_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 2
        ite_num.append(value[0])  # 5
        ite_proportion1.append(value[1])
        i += 1

# matplot_three(ci_seed, Y, ci, seed, "counter instance & seed instance(based on proportion)", "CI/seed", "CI", "seed instance", "feature_set", "#instance")

# matplot_two(ci_seed, ite_num, "ci_seed VS iteration num", "ci_seed", "#iteration", "feature_set", "perturbation num", savefile)
#
# matplot_two(ite_num, ci_seed, "iteration num VS ci_seed", "#iteration", "ci_seed", "feature_set", "ci_seed proportion", savefile)


# accuracy_file = "../result_file/k=4/accuracy"
accuracy_file = "../result/census/k2(depth=k+1)/accuracy"
accuracy = []
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 2
        accuracy.append(value[0]/100.0)  # 5
        i += 1

accuracy_file1 = "../result/census/k=2/accuracy"
accuracy1 = []
with open(accuracy_file1, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 2
        accuracy1.append(value[0]/100.0)  # 5
        i += 1

matplot_two(accuracy, accuracy1, "two accuracy", "accuracy", "accuracy(k=5)", "feature_set", "accuracy(%)")

print("-->accuracy for depth = k + 1")
print(max(accuracy))
print(max(accuracy1))


# matplot_two(accuracy, ite_num, "accuracy VS iteration num", "accuracy", "#iteration", "feature_set", "perturbation num", savefile)
#
# matplot_two(accuracy, ci_seed, "accuracy VS ci_seed", "accuracy", "CI/seed", "feature_set", "CI proportion", savefile)

# matplot_four(accuracy, Y, ci_seed, ci, seed, "accuracy VS four", "accuracy", "ci_seed", "CI", "seed", "feature set", "ci_seed")



# plt.scatter(Y, ite_num, label = "iteration num")
# plt.title("iteration num")
# plt.xlabel("feature_set")
# plt.ylabel("#iteration")
# plt.legend()
# plt.show()
#
# plt.scatter(Y, seed, label = "seed")
# plt.title("seed")
# plt.xlabel("feature_set")
# plt.ylabel("seed")
# plt.legend()
# plt.show()
#
# plt.scatter(Y, ci, label = "CI")
# plt.title("CI")
# plt.xlabel("feature_set")
# plt.ylabel("CI")
# plt.legend()
# plt.show()
#
# plt.scatter(Y, ci_seed, label = "ci_seed")
# plt.title("ci_seed")
# plt.xlabel("feature_set")
# plt.ylabel("ci_seed")
# plt.legend()
# plt.show()











'''
# 3 accuracy based on accuracy
dictionary = dict(zip(Y, X))
print(dictionary)
print(len(dictionary))
print("first sort", (sorted(dictionary.items(), key=lambda d: d[0])))
sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
print("second sort", sorted_dict)
print(len(sorted_dict))

sorted_feature = []
sorted_x1 = []
sorted_x2 = []
for dicts in sorted_dict:
    sorted_feature.append(dicts[0])
    sorted_x1.append(x1[dicts[0]])
    sorted_x2.append(x2[dicts[0]])
print(sorted_feature)
print(sorted_x1)
print(sorted_x2)

plt.title("accuracy")
sorted_items = []
for dicts in sorted_dict:
    item = dicts[1]
    sorted_items.append(float(item))
print("sorted_items:", sorted_items)
print(Y)
print("length of sorted_items:", len(sorted_items))
plt.plot(Y, sorted_items, label="accuracy")
plt.plot(Y, sorted_x1, label="label 0's accuracy")
plt.plot(Y, sorted_x2, label="label 1's accuracy")
plt.legend()
plt.xlabel("feature_set")
plt.ylabel("accuracy")
# plt.savefig("../figures/census/k=6/three accuracy(base accuracy).eps")
plt.show()


# plt.plot(Y, sorted_items)
# plt.title("accuracy (sorted)")
# plt.xlabel("feature_set")
# plt.ylabel("accuracy(%)")
# plt.savefig("../figures/census/k=6/average_accuracy(sorted)1.eps")
# plt.show()
# get_figure(X, Y, True, "accuracy (sorted)", "featrue_set", "accuracy(%)", "../figures/census/k=6/average_accuracy(sorted).eps")


# 3 accuracy based on label 0's accuracy
dictionary = dict(zip(Y, x1))
print(dictionary)
print(len(dictionary))
# print("first sort", (sorted(dictionary.items(), key=lambda d: d[0])))
sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
print("second sort", sorted_dict)
print(len(sorted_dict))

sorted_feature = []
sorted_x = []
sorted_x2 = []
for dicts in sorted_dict:
    sorted_feature.append(dicts[0])
    sorted_x.append(X[dicts[0]])
    sorted_x2.append(x2[dicts[0]])
print(sorted_feature)
print(sorted_x)
print(sorted_x2)

plt.title("accuracy(based on label 0's accuracy)")
sorted_items = []
for dicts in sorted_dict:
    item = dicts[1]
    sorted_items.append(float(item))
print("sorted_items:", sorted_items)
print(Y)

plt.plot(Y, sorted_items, label="label 0's accuracy")
plt.plot(Y, sorted_x, label="accuracy")
plt.plot(Y, sorted_x2, label="label 1's accuracy")
plt.legend()
plt.xlabel("feature_set")
plt.ylabel("accuracy")
# plt.savefig("../figures/census/k=6/three accuracy(base 0's accuracy).eps")
plt.show()














accuracy_file = '../results/accuracy4'
X4, Y4 = [], []
x14, x24 = [], []
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    max_accuracy = 0.0
    min_accuracy = 100.0
    max_accuracy_0 = 0.0
    min_accuracy_0 = 100.0
    max_accuracy_1 = 0.0
    min_accuracy_1 = 100.0
    min_feature_set = feature_sets[0]
    for line in lines:  # 3
        value = [float(s) for s in line.split(" ")]  # 4
        X4.append(value[0]/100.0)
        x14.append(value[1]/100.0)
        x24.append(value[2]/100.0)
        Y4.append(i)
        i += 1
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            min_feature_set = feature_sets[i]
        if value[0] < min_accuracy:
            min_accuracy = value[0]
            max_feature_set = feature_sets[i]
        if value[1] > max_accuracy_0:
            max_accuracy_0 = value[1]
            min_feature_set_0 = feature_sets[i]
        if value[1] < min_accuracy_0:
            min_accuracy_0 = value[1]
            max_feature_set_0 = feature_sets[i]
        if value[2] > max_accuracy_1:
            max_accuracy_1 = value[2]
            min_feature_set_1 = feature_sets[i]
        if value[2] < min_accuracy_1:
            min_accuracy_1 = value[2]
            max_feature_set_1 = feature_sets[i]


accuracy_file = '../results/accuracy5'
X5, Y5 = [], []
x15, x25 = [], []
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    max_accuracy = 0.0
    min_accuracy = 100.0
    max_accuracy_0 = 0.0
    min_accuracy_0 = 100.0
    max_accuracy_1 = 0.0
    min_accuracy_1 = 100.0
    min_feature_set = feature_sets[0]
    for line in lines:  # 3
        value = [float(s) for s in line.split(" ")]  # 4
        X5.append(value[0]/100.0)
        x15.append(value[1]/100.0)
        x25.append(value[2]/100.0)
        Y5.append(i)
        i += 1
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            min_feature_set = feature_sets[i]
        if value[0] < min_accuracy:
            min_accuracy = value[0]
            max_feature_set = feature_sets[i]
        if value[1] > max_accuracy_0:
            max_accuracy_0 = value[1]
            min_feature_set_0 = feature_sets[i]
        if value[1] < min_accuracy_0:
            min_accuracy_0 = value[1]
            max_feature_set_0 = feature_sets[i]
        if value[2] > max_accuracy_1:
            max_accuracy_1 = value[2]
            min_feature_set_1 = feature_sets[i]
        if value[2] < min_accuracy_1:
            min_accuracy_1 = value[2]
            max_feature_set_1 = feature_sets[i]


accuracy_file = '../results/accuracy1'
X1, Y1 = [], []
x11, x21 = [], []
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    max_accuracy = 0.0
    min_accuracy = 100.0
    max_accuracy_0 = 0.0
    min_accuracy_0 = 100.0
    max_accuracy_1 = 0.0
    min_accuracy_1 = 100.0
    min_feature_set = feature_sets[0]
    for line in lines:  # 3
        value = [float(s) for s in line.split(" ")]  # 4
        X1.append(value[0]/100.0)
        x11.append(value[1]/100.0)
        x21.append(value[2]/100.0)
        Y1.append(i)
        i += 1
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            min_feature_set = feature_sets[i]
        if value[0] < min_accuracy:
            min_accuracy = value[0]
            max_feature_set = feature_sets[i]
        if value[1] > max_accuracy_0:
            max_accuracy_0 = value[1]
            min_feature_set_0 = feature_sets[i]
        if value[1] < min_accuracy_0:
            min_accuracy_0 = value[1]
            max_feature_set_0 = feature_sets[i]
        if value[2] > max_accuracy_1:
            max_accuracy_1 = value[2]
            min_feature_set_1 = feature_sets[i]
        if value[2] < min_accuracy_1:
            min_accuracy_1 = value[2]
            max_feature_set_1 = feature_sets[i]


dictionary6 = dict(zip(Y, X))
length = len(dictionary6)
Y6 = list(range(0, length))
# sorted_dict6 = sorted(dictionary6.items(), key=lambda d: d[0])
sorted_dict6 = sorted(dictionary6.items(), key=lambda x: x[1])
print("6sorted_dict:", sorted_dict)
sorted_items6 = []
for dicts in sorted_dict6:
    item = dicts[1]
    sorted_items6.append(float(item))

dictionary4 = dict(zip(X4, Y4))
length = len(dictionary4)
Y4 = list(range(0, length))
sorted_dict4 = sorted(dictionary4.items(), key=lambda d: d[0])
# print("sorted_dict:", sorted_dict)
sorted_items4 = []
for dicts in sorted_dict4:
    item = dicts[0]
    sorted_items4.append(float(item))

dictionary5 = dict(zip(X5, Y5))
length = len(dictionary5)
Y5 = list(range(0, length))
sorted_dict5 = sorted(dictionary5.items(), key=lambda d: d[0])
# print("sorted_dict:", sorted_dict)
sorted_items5 = []
for dicts in sorted_dict5:
    item = dicts[0]
    sorted_items5.append(float(item))

dictionary3 = dict(zip(X1, Y1))
length = len(dictionary3)
Y3 = list(range(0, length))
sorted_dict3 = sorted(dictionary3.items(), key=lambda d: d[0])
# print("sorted_dict:", sorted_dict)
sorted_items3 = []
for dicts in sorted_dict3:
    item = dicts[0]
    sorted_items3.append(float(item))


print(len(Y), len(Y4), len(Y5))

# print("sorted:", sorted_items)
print(len(sorted_items))
plt.title("accuracys")
plt.plot(Y3, sorted_items3, label="k=3")
plt.plot(Y4, sorted_items4, label="k=4")
plt.plot(Y5, sorted_items5, label="k=5")
plt.plot(Y6, sorted_items6, label="k=6")
# plt.scatter(Y, sorted_items)
plt.xlabel("feature sets")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("../figures/census/multiple_k.eps")
plt.show()

#
# plt.title("accuracy3")
# plt.plot(Y, sorted_items)
# # plt.scatter(Y, sorted_items)
# plt.xlabel("feature sets")
# plt.ylabel("accuracy")
# plt.legend()
# # plt.savefig(savefile)
# plt.show()
#
# plt.title("accuracy4")
# plt.plot(Y4, sorted_items4)
# # plt.scatter(Y, sorted_items)
# plt.xlabel("feature sets")
# plt.ylabel("accuracy")
# plt.legend()
# # plt.savefig(savefile)
# plt.show()

'''

















