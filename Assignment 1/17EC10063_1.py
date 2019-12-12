# 17EC10063
# Kshitij Agrawal
# Assignment 1 (Decision Tree)

# To run : python3 17EC10063_a1.py
# (Won't work with python2, so please ensure to explicitly mention python3)

# To predict : Change the value of the features in the 'test()' function (Line no 180) 
# (Currently a sample data is fed, one may change the values as needed)

import math
import numpy as np

filename = "data1_19.csv"

class DecisionTree:
    '''
        DecisionTree Class wraps all functions from extraction of data to building and finally printing the built tree
        '''
    def __init__(self):
        # Initializes some important parameters of the logic (the dataset and the feature list)
        self.data = {}
        self.features = []
        self.tree = {}
        # Extracts data and stores in a form easy to use (dictionaries having labels as keys and list as values)
        self.extract_data()
        # Calls the build_tree function to obtain the needed tree
        self.tree = self.build_tree(self.features[-1], self.data, self.features)
        # Prints the tree as required 
        print("\n\tBUILT TREE\n")
        self.print_tree(self.tree)
        # To predict, go to test() (Line 180)
        print("\n\tPREDICTIONS\n")
        self.test()

    def get_data(self):
        with open(filename, 'r') as file:
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

    def get_root(self, data, features):
        '''
        Uses Information gain to choose the best node to select as the current root
            and returns the label corresponding to it
            '''
        info_gain = {}
        for feature in features[:-1]:
            info_gain[feature] = self.information_gain(data, feature, features[-1])

        return max(info_gain, key=info_gain.get)

    def get_new_data(self, root, value, dataset, features):
        '''
            Constructs a new copy of the dataset with the rows having the current chosen node as value
            '''
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

    def get_actual_value(self, test_data):

        '''  Searches the data set to find the actual value for a given test_data   '''
        '''  Returns the most frequent result in case of more than one outcome   '''

        count = len(self.data['survived'])

        vals = []

        for i in range(count):
            all_taken = 1
            for label in self.features[:-1]:
                if self.data[label][i] != test_data[label]:
                    all_taken = 0
            if all_taken:
                vals.append(self.data['survived'][i])
        return max(vals, key=vals.count)

    def test(self):
        ''' Predicts the output of the decision tree on custom inputs'''

        test_data = {'pclass' : 'crew', 'age' : 'adult', 'gender' : 'male'}

        # Set custom values for the three labels of the test_data (Please ensure data entered is in the proper format)
        # Change the values as needed

        print('   Current sample = ',end='')
        print(test_data)
        print('   Output predicted for the current sample (Survived or not)', end=' = ')
        result = self.predict(test_data, self.tree)     #   Predicted output
        print(result)

        print('   Actual output for the current sample (Survived or not)', end=' = ')
        result = self.get_actual_value(test_data)
        print(result, end='\n\n')                                   #   Actual output

    def predict(self, test, tree):
        '''
            Call this function to run the code on Custom data, 
                the function recursively searches in the tree and returns the result as soon as it hits a leaf (Yes/ No)
            '''
        for node in tree.keys():
            value = test[node]
            tree = tree[node][value]

            if type(tree) is dict:      #   Indicates the current node is not a leaf, and has further branches
                return self.predict(test, tree)     #   Recursively check for other branches in the subtree of current node
            else:   # Reached a leaf, return
                return tree

    def print_tree(self, tree, tabs = 0):
        '''
            Recursively prints the nodes and their branches in the required format
            '''
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
    # Tree is an instance of the DecisionTree class, calls the __init__ method and progressively builds the tree
    Tree = DecisionTree()   
