import itertools
import numpy as np

# Print a decision tree
def print_tree(node, depth=0):
    # print("node:", node)
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))

direction = "../result/census/k=3/"

# time_file = direction + "/time"
# DT_time1 = []
# DT_time2 = []
# min_DT_time = 100
# min_DT_feature = 0
# with open(time_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         DT_time1.append(value[0])  # 5
#         DT_time2.append(value[1])
#         if value[0] < min_DT_time:
#             min_DT_time = value[0]
#             min_DT_feature = i
#         i += 1

    # decision_tree_file = "../result_file/k=3/DT_trees"
    #
    # all_DT_trees = []
    # with open(decision_tree_file, 'r') as f:
    #     for line in f:
    #         all_DT_trees.append(line.split("\n")[0])
    # print("-->decision_tree_list:", all_DT_trees)


# time_file = direction + "/time"
# per_time = []
# min_per_time = 100
# min_per_feature = 0
# with open(time_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         per_time.append(value[0])  # 5
#         if value[0] < min_per_time:
#             min_per_time = value[0]
#             min_per_feature = i
#         i += 1
#
# print("---->time")
# print(min(DT_time1))
# print(min(DT_time2))
# # print(min_DT_feature)
# # print(per_time[min_DT_feature])
#
# print("")
# print(min_per_time)
# print(min_per_feature)
# # print(DT_time1[min_per_feature])
# # print(DT_time2[min_per_feature])


accuracy_file = direction + "/accuracy"
accuracy = []
accuracy_0 = []
accuracy_1 = []
max_accuracy_feature = 0
max_accuracy = 0
with open(accuracy_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 2
        accuracy.append(value[0])  # 5
        accuracy_0.append(value[1])
        accuracy_1.append(value[2])
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            max_accuracy_feature = i
        i += 1

print("-->accuracy")
print(max_accuracy)
print(accuracy_0[max_accuracy_feature])
print(accuracy_1[max_accuracy_feature])
print("-->max_accuracy_feature", max_accuracy_feature)
# all_feature_set = list(range(1, 13 + 1))
# feature_sets = list(itertools.combinations(all_feature_set, 2))
# print("-->feature set:", feature_sets[max_accuracy_feature])

time_file = direction + "/time"
DT_time1 = []
DT_time2 = []
min_DT_time = 100
min_DT_feature = 0
with open(time_file, 'r') as f:  # 1
    lines = f.readlines()  # 2
    i = 0
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 2
        DT_time1.append(value[0])  # 5
        DT_time2.append(value[1])
        if value[0] < min_DT_time:
            min_DT_time = value[0]
            min_DT_feature = i
        i += 1

print("-->time:")
print("average time of learning one decision tree:", np.mean(DT_time1))
print("average time of calculating accuracy:", np.mean(DT_time2))



tree = {'index': 1, 'right': {'index': 0, 'right': {'index': 0, 'right': 0, 'value': 11.0, 'left': 0}, 'value': 10.0, 'left': {'index': 0, 'right': 0, 'value': 1.0, 'left': 0}}, 'value': 1.0, 'left': {'index': 0, 'right': {'index': 0, 'right': 1, 'value': 10.0, 'left': 0}, 'value': 1.0, 'left': {'index': 2, 'right': 1, 'value': 1.0, 'left': 1}}}
print_tree(tree)


#
# direction2 = "../retrained_result/census/k=3(all)"
# accuracy_file = direction + "/accuracy"
# accuracy = []
# accuracy_0 = []
# accuracy_1 = []
# max_accuracy_feature = 0
# max_accuracy = 0
# with open(accuracy_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         accuracy.append(value[0])  # 5
#         accuracy_0.append(value[1])
#         accuracy_1.append(value[2])
#         if value[0] > max_accuracy:
#             max_accuracy = value[0]
#             max_accuracy_feature = i
#         i += 1
#
# print("-->accuracy")
# print(max_accuracy)
# print(accuracy_0[max_accuracy_feature])
# print(accuracy_1[max_accuracy_feature])
# print("-->max_accuracy_feature", max_accuracy_feature)
# all_feature_set = list(range(1, 13 + 1))
# feature_sets = list(itertools.combinations(all_feature_set, 3))
# print("-->feature set:", feature_sets[max_accuracy_feature])


# all_feature_set = list(range(1, 13 + 1))
# feature_sets = list(itertools.combinations(all_feature_set, 3))
# print("-->feature set:", feature_sets[max_accuracy_feature])

# all_feature_set = list(range(1, 16 + 1))
# feature_sets = list(itertools.combinations(all_feature_set, 2))
# print("-->feature_set", feature_sets[max_accuracy_feature])


# DT_file = direction + "/DT_trees"
# all_DT_trees = []
# with open(DT_file, 'r') as f:  # 1
#     for line in f:
#         all_DT_trees.append(line.split("\n")[0])
# print("-->tree", all_DT_trees[max_accuracy_feature])
# tree = all_DT_trees[max_accuracy_feature]
tree = {'index': 1, 'right': {'index': 0, 'right': 1.0, 'value': 14.0, 'left': 0.0}, 'value': 3.0, 'left': {'index': 0, 'right': 1.0, 'value': 90.0, 'left': 0.0}}
print_tree(tree)


# ci_seed_file = direction + "/CI_num"
# ci_seed = []
# min_ci_seed = 1
# min_ci_seed_feature = 0
# max_ci_seed = 0
# max_ci_seed_feature = 0
# with open(ci_seed_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         ci_seed.append(value[2])  # 5
#         if value[2] < min_ci_seed:
#             min_ci_seed = value[2]
#             min_ci_seed_feature = i
#         if value[2] > max_ci_seed:
#             max_ci_seed = value[2]
#             max_ci_seed_feature = i
#         i += 1
#
# iteration_file = direction + "/iteration_num"
# itr_num = []
# max_itr_num = 0
# max_itr_num_feature = 0
# min_itr_num = 100
# min_itr_num_feature = 0
# with open(iteration_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         ci_seed.append(value[0])  # 5
#         if value[0] < min_itr_num:
#             min_itr_num = value[0]
#             min_itr_num_feature = i
#         if value[0] > max_itr_num:
#             max_itr_num = value[0]
#             max_itr_num_feature = i
#         i += 1
#
# print("---->ci_seed")
# print(min_ci_seed)
# print(min_ci_seed_feature)
#
# # print("")
# # print(max_ci_seed)
# # print(max_ci_seed_feature)
#
#
# print("---->iteration number")
# print(max_itr_num)
# print(max_itr_num_feature)
# print(ci_seed[max_itr_num_feature])

# print("")
# print(min_itr_num)
# print(min_itr_num_feature)
# print(ci_seed[min_itr_num_feature])


# time_file = "../result/census/k=2/time"
# DT_time1 = []
# DT_time2 = []
# min_DT_time = 100
# min_DT_feature = 0
# with open(time_file, 'r') as f:  # 1
#     lines = f.readlines()  # 2
#     i = 0
#     for line in lines:  # 3
#         value = [float(s) for s in line.split()]  # 2
#         DT_time1.append(value[0])  # 5
#         DT_time2.append(value[1])
#         if value[0] < min_DT_time:
#             min_DT_time = value[0]
#             min_DT_feature = i
#         i += 1
#


# print("-->tunning time")
# run_time = sum(DT_time1)
# run_time += sum(DT_time2)
# print(run_time)

# all_feature_set = list(range(1, 16 + 1))
# feature_sets = list(itertools.combinations(all_feature_set, 5))
# print("-->feature_set", len(feature_sets))
# index = 0
# for f in feature_sets:
#     if f == (2, 4, 7, 13, 15):
#         print("-->index", index)
#     index += 1
#
# print(feature_sets[1797])
# new_feature_lists = feature_sets[1798:]
# print(new_feature_lists[0])
