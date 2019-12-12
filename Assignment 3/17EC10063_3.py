# 17EC10063
# Kshitij Agrawal
# Assignment 3 (Adaboost classifier)

# To run : python3 17EC10063_3.py
# (Won't work with python2, so please ensure to explicitly mention python3)

import math
import numpy as np

train_file = "data3_19.csv"
test_file = "test3_19.csv"

class Adaboost:
    ''' Wrapper over all necessary member variables and functions'''
    def __init__(self, rounds = 3):
        self.rounds = rounds
        self.inst = Decision_Tree()
        self.clf = self.inst.tree
        self.clfs = []
        self.master_record = self.inst.data
        self.tree_data = self.master_record
        self.n_samples = self.inst.n_samples
        self.indices = [idx for idx in range(self.n_samples)]
        self.wts = [1 / self.n_samples for idx in range(self.n_samples)]
        self.preprocess()
        self.fit(self.clf)
        self.predict()

    def preprocess(self):
        ''' Indexes the dataset for random sampling based on weights during boosting'''
        indexed_data = []
        for n in range(self.n_samples):
            curr_sample = {}
            for key in self.master_record.keys():
                curr_sample[key] = self.master_record[key][n]
            indexed_data.append(curr_sample)
        self.master_record = indexed_data

    def resample(self):
        ''' Contructs a new data set according to the updated weights of the samples'''
        new_data = []
        for n in range(self.n_samples):
            idx = self.indices[n]
            curr_sample = self.master_record[idx]
            new_data.append(curr_sample)
        self.tree_data = {}
        for key in new_data[0].keys():
            self.tree_data[key] = []

        for n in range(self.n_samples):
            for key in self.tree_data.keys():
                self.tree_data[key].append(new_data[n][key])
        return new_data

    def update_wts(self, match):
        ''' Boosts weights of misclassified examples while reduces the one for the correct ones'''
        correct = sum(match)
        error = 0
        for n in range(self.n_samples):
            idx = self.indices[n]
            if not match[n]:
                error += self.wts[idx]
        if error > 0.5:
            error = 1 - error
        alpha = 0.5 * np.log((1.0 - error) / (error + 1e-6))
        copy = self.wts
        for n in range(self.n_samples):
            idx = self.indices[n]
            old_wt = self.wts[idx]
            new_wt = 0
            if match[n]:
                new_wt = old_wt * np.exp(-alpha)
            else:
                new_wt = old_wt * np.exp(alpha)
            copy[idx] = new_wt
        total = sum(copy)

        self.wts = [x / total for x in copy]
        return alpha, self.n_samples - correct

    def fit(self, clf):
        ''' Runs the Adaboost classifier for 3 rounds'''
        # np.random.seed(2)
        for itr in range(self.rounds):
            print("Boosting Round {}: ".format(itr + 1))
            print("    ",end='')
            if itr:
                self.indices = list(np.random.choice(self.indices, self.n_samples, self.wts))
            p = self.inst.accuracy(clf, self.resample())
            p = [int(x) for x in p]
            alpha, error = self.update_wts(p)
            now = (clf, alpha)
            self.clfs.append(now)
            # self.inst.print_tree(clf)
            print("Significance(α) = {}\n    Misclassified samples = {} / {}".format(alpha, error, self.n_samples))
            clf = self.inst.build(self.tree_data)
        print("\n    Training on test set...")

    def get_data(self):
        ''' Extracts test set for prediction'''
        test = {}
        with open(test_file, 'r') as file:
            data = file.read()
            df = data.split('\n')
            df = df[:-1]
            df = [vals.split(',') for vals in df]
            labels = self.inst.features
            for i in range(len(labels)):
                label = labels[i]
                test[label] = [v[i] for v in df]
        
        test_set = {}
        label = ''
        for key in test.keys():
            if key != 'survived':
                label = key
                test_set[key] = test[key]
        dummy = []

        total_size = len(test_set[label])
        for idx in range(total_size):
            temp = {}
            for key in test_set.keys():
                temp[key] = test_set[key][idx]
            dummy.append(temp)

        test_set = dummy
        return test_set, [x for x in test['survived']]

    def predict(self):
        ''' Predicts the samples on the test set using the classifiers stored in each round of boosting'''
        test_set, outcome = self.get_data()
        sample_size = len(test_set)
        correct = 0
        idx = 0
        org = 0
        for test in test_set:
            sum_res = 0
            l = []
            i = 0
            now = 0
            for clf in self.clfs:
                sig = clf[1]
                classifier = clf[0]
                result = self.inst.get(test, classifier)
                add = result
                l.append(add)
                if add == 'yes':
                    sum_res += sig
                else:
                    sum_res -= sig
                if i == 0:
                    now = add
                i += 1
            ans = ''
            if sum_res < 0:
                ans = 'no'
            else:
                ans = 'yes'
            correct += (ans == outcome[idx])
            org += (now == outcome[idx])
            idx += 1
        accuracy = correct / sample_size * 100
        accuracy = round(accuracy, 2)
        print("      Accuracy on test set = ", end='')
        print(accuracy, end='')
        print(" %")

class Decision_Tree:
    ''' DecisionTree Class wraps all functions from extraction of data to building and finally printing the built tree '''
    def __init__(self):
        # Initializes some important parameters of the logic (the dataset and the feature list)
        self.data = {}
        self.features = []
        self.tree = {}
        self.n_samples = 0
        self.extract_data()
        self.tree = self.build(self.data)

    def get_data(self):
        with open(train_file, 'r') as file:
            data = file.read()
            return data

    def extract_data(self):
        data_frame = self.get_data()
        data_frame = data_frame.split('\n')
        data_frame = data_frame[:-1]
        data_frame = [vals.split(',') for vals in data_frame]
        labels = data_frame[0]
        self.features = labels
        data_frame = data_frame[1:]         # Remove the labels' column
        for i in range(len(labels)):
            self.data[labels[i]] = [vals[i] for vals in data_frame]
        self.n_samples = len(self.data[labels[0]])

    def build(self, data):    # A wrapper over the build_tree function
        return self.build_tree(self.features[-1], data, self.features)

    def get_root(self, data, features):
        ''' Uses Information gain to choose the best node to select as the current root
            and returns the label corresponding to it '''
        info_gain = {}
        for feature in features[:-1]:
            info_gain[feature] = self.information_gain(data, feature, features[-1])

        return max(info_gain, key=info_gain.get)

    def get_new_data(self, root, value, dataset, features):
        ''' Constructs a new copy of the dataset with the rows having the current chosen node as value '''
        new_data = {}

        for i in range(len(features)):
            new_data[features[i]] = []
            for j in range(len(dataset[root])):
                if dataset[root][j] == value:
                    new_data[features[i]].append(dataset[features[i]][j])
        return new_data

    def build_tree(self, target, dataset, features):
        '''
            The heart of the logic, recursively builds the tree using information gain as the criteria for choosing
                the nodes, and calling the function on the values of the chosen feature
            '''

        vals = np.unique(dataset[target])

        if len(vals) == 1:      #   If total number of all the values of the current label is 1 (either all Yes or No), stop recursion, add leaf and return
            return vals[0]
        ''' Only one feature (the final labels (Yes / No)) remain, 
            implying all features have been used, return the label with the max frequency in the current node '''

        if(len(features) == 1):     
            return max(dataset[features[0]], key=dataset[features[0]].count)

        root = self.get_root(dataset, features)

        tree_node = {root: {}}

        new_features = [feature for feature in features if feature != root]     # Remove the feature corresponding to the root chosen

        branches = np.unique(dataset[root])

        for value in branches:
            new_data = self.get_new_data(root, value, dataset, features)
            subtree = self.build_tree(target, new_data, new_features)   # Recursively call the build_tree function to grow the tree
            tree_node[root][value] = subtree

        return tree_node

    # Some helper functions which return the entropy and Information gain for different features 

    def entropy(self, data, node):
        total = len(data[node])
        S = 0
        positive = data[node].count('yes')
        negative = data[node].count('no')
        positive /= total
        negative /= total
        if positive:
            S += positive * math.log2(positive)
        if negative:
            S += negative * math.log2(negative)
        S *= -1
        return S

    def entropy_branches(self, data, node, target):
        labels = np.unique(data[node])
        weights = 0
        total = len(data[node])
        counts = {}
        freq = {}
        for label in labels:
            counts[label] = []
            counts[label].append(0)
            counts[label].append(0)
            freq[label] = 0
        for i in range(len(data[node])):
            label = data[node][i]
            freq[label] += 1
            if data[target][i] == 'yes':
                counts[label][0] += 1
            else:
                counts[label][1] += 1
        for label in labels:
            pos = counts[label][0]
            neg = counts[label][1]
            cnt = freq[label]
            pos /= cnt
            neg /= cnt
            now = 0
            if pos:
                now += pos * math.log2(pos)
            if neg:
                now += neg * math.log2(neg)
            now *= (cnt / total)
            now *= -1
            weights += now
        return weights

    def information_gain(self, data, attribute, target):
        total = self.entropy(data, target)
        weights = self.entropy_branches(data, attribute, target)
        return total - weights

    def accuracy(self, tree, test_set):
        data = []
        for n in range(len(test_set)):
            curr_sample = test_set[n]
            new_sample = {}
            for key in curr_sample.keys():
                if key != 'survived':
                    new_sample[key] = curr_sample[key]
            data.append(new_sample)
        prediction = self.predict(data, tree)
        match = []
        for n in range(len(test_set)):  
            match.append(prediction[n] == test_set[n]['survived'])
        return match
    
    def predict(self, test_set, tree):
        result = []
        for test in test_set:
            result.append(self.get(test, tree))
        return result
    
    def get(self, test, tree):
        for node in tree.keys():
            value = test[node]
            tree = tree[node][value]
            if type(tree) is dict:
                return self.get(test, tree)
            return tree

    def print_tree(self, tree, tabs = 0):
        ''' Recursively prints the nodes and their branches in the required format '''
        for node in tree.keys():
            for vals in tree[node]:
                print('  ',end='')
                for i in range(tabs):
                    print('   ', end = '')
                if type(tree[node][vals]) is dict:
                    print(node, end = ' = ')
                    print(vals, end = ':\n')
                    self.print_tree(tree[node][vals], tabs + 1)
                else:
                    print(node, end = ' = ')
                    print(vals, end = ': ')
                    print(tree[node][vals])

if __name__ == "__main__":
    Adaboost()