
import numpy as np
import csv
import pandas as pd


#Evaluates a sample using the given decision tree dt. Returns true if the decision tree gets it right, false if not
def evalSample(dt, sample):   
    while (len(dt) == 2):
        dt = dt[1][sample[dt[0]]]
    return (dt[0] == sample['class'])

mushroomTrain = pd.read_csv("mushrooms_train_updated.csv") #training set
mushroomTest = pd.read_csv("mushrooms_test_updated.csv") #testing set

#Just a dummy assingment
df = mushroomTrain 
#Dummy Decision Tree, ID3 should create your own decision tree.This is a nested representation. Each node is a list. 
#if the node has one entry then it is a leaf node and performs the classification. Otherwise the node has two entries:
#the first entry is the feature ('bruises') and the second entry is a dictionary containing the tree below this node
#for example, bruises has two different possible values ('f' and 't'), therefore the dictionary has those two values
#'t' goes to a leaf node classifying 'p'. However, the 'f' branch has more decision making in this case 'gill-spacing' 
#that all lead to terminal nodes.  
dt = ['bruises', {'f': ['gill-spacing', {'c': ['e'], 
                                         'w': ['p'], 
                                         'd': ['e']}], 
                  't': ['p']}] 
 
print(evalSample(dt, df.iloc[7])) 

minEntropy = float("inf") #Variable to keep track of lowest entropy found
bestFeature = "" #Variable to keep track of feature to give the lowest conditional entropy


#Your ID3 algorithm goes here! Use evalSample to evaluate your completed tree on both training and testing sets.

#For calculating entropy
def cal_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0

#calculating feature gain
def cal_featuregain(data, classes, feature):
    gain = 0
    num_data = len(data)

    # computing all the different column features
    fs = {}
    for row in data:
        if row[feature] not in fs.keys():
            fs[row[feature]] = 1
        else:
            fs[row[feature]] += 1

    for fi in fs.keys():
        f_entropy = 0
        row_index = 0
        new_class = {}
        class_count = 0
        for row in data:
            if row[feature] == fi:
                class_count += 1
                if classes[row_index] in new_class.keys():
                    new_class[classes[row_index]] += 1
                else:
                    new_class[classes[row_index]] = 1
            row_index += 1

        for aclass in new_class.keys():
            f_entropy += cal_entropy(float(new_class[aclass]) / class_count)

        gain += float(fs[fi]) / num_data * f_entropy
    return gain


def cal_total_entropy(targets):
    d_targets = {}
    n = len(targets)
    for t in targets:
        if t not in d_targets:
            d_targets[t] = 1
        else:
            d_targets[t] += 1
    entropy = 0
    for t in d_targets:
        p = d_targets[t] / float(n)
        entropy += cal_entropy(p)

    return entropy


def sub_data(data, targets, feature, fi):
    new_data = []
    new_targets = []
    num_features = len(data[0])
    row_index = 0
    for row in data:
        if row[feature] == fi:
            if feature == 0:
                new_row = row[1:]
            elif feature == num_features:
                new_row = row[:-1]
            else:
                new_row = row[:feature]
                new_row.extend(row[feature + 1:])

            new_data.append(new_row)
            new_targets.append(targets[row_index])
        row_index += 1

    return new_targets, new_data



def make_tree(data, classes, features):
    
    num_data = len(data)
    num_features = len(features)
    
    uniqueT = {}
    for aclass in classes:
        if aclass in uniqueT.keys():
            uniqueT[aclass] += 1
        else:
            uniqueT[aclass] = 1

    default = max(uniqueT, key=uniqueT.get)
    if num_data == 0 or num_features == 0:
        return default
    elif len(np.unique(classes)) == 1:
        
        return classes[0]
    else:
        
        totalEntropy = cal_total_entropy(classes)
        gain = np.zeros(num_features)
        for feature in range(num_features):
            g = calc_feature_gain(data, classes, feature)
            gain[feature] = totalEntropy - g
        best = np.argmax(gain)  
        fi_s = np.unique(np.transpose(data)[best])
        feature = features.pop(best)    
        tree = {feature: {}}
        
        for fi in fi_s:
           
            t, d = sub_data(data, classes, best, fi)
           
            subtree = make_tree(d, t, features)
           
            tree[feature][fi] = subtree
        return tree



#printing decision tree
def print_tree(tree, answer):
    
    tree_T = {}
    for root in tree.keys():
        level = []
        leaves = {}
        next_keys = []
        for S in tree[root]:
            if type(tree[root][S]) is not dict:
                leaf = tree[root][S]
                if leaf in leaves:
                    leaves[leaf].append(S)
                else:
                    leaves[leaf] = [S]
            else:
                next_keys.append(S)
                print_tree(tree[root][S], S)

        if len(next_keys) != 0:
            level.extend(next_keys)
        level.append(leaves)
        tree_T[root] = level
        print(answer, tree_T)


#training decision tree
def training(csv_file):
    data, targets, features = import_data(filename=csv_file)
    tree = make_tree(data, targets, features)
    return tree


def tree_output(tree, data_row, features):
    if type(tree) is not dict:
        return str(tree)
    else:
        for key in tree.keys():
            f_index = features.index(key)
            f_i = data_row[f_index]
            return tree_output(tree[key][f_i], data_row, features)


def validator(filename, tree):

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)
    features = raw_data.pop(0)
    features.remove("class")
    targets = [row.pop(0) for row in raw_data]

    gd = 0
    n = len(raw_data)
    row_index = 0
    for row in raw_data:
        output = tree_output(tree, row, features)
        if output == targets[row_index]:
            gd += 1
        row_index += 1
    
    return gd/float(n)

        

#importing data
def import_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)
    features = raw_data.pop(0)
    features.remove("class")
    targets = [row.pop(0) for row in raw_data]

    test_1 = len(raw_data[0]) == len(features)
    test_2 = len(raw_data) == len(targets)
    if test_1 and test_2:
        return raw_data, targets, features
    else:
        print("ERROR")
        exit(-1)


def main():
    train_file = "mushrooms_train_updated.csv"
    test_file = "mushrooms_test_updated.csv"

    decision_tree = training(train_file)
    per_correct = validator(test_file, decision_tree)
    print("____________________________")
    print_tree(decision_tree, 'root')
    print("____________________________")
    print("percent of correct classifications:", per_correct)


main()