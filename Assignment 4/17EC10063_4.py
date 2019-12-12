# 17EC10063
# Kshitij Agrawal
# Assignment 4 (K-Means Clustering)

# To run : python3 17EC10063_4.py
# (Won't work with python2, so please ensure to explicitly mention python3)

import numpy as np

train_file = "data4_19.csv"

class K_means:
    def __init__(self):
        self.k = 3
        self.iterations = 10
        self.features = ["sepal length", "sepal width", "petal length", "petal width", "name"]
        self.data = []
        self.truth_vals = set()
        self.class_cnt = {}
        self.preprocess()
        self.clusters, self.means = self.train()
        self.Jaccard()

    def get_data(self):
        with open(train_file, 'r') as file:
            data = file.read()
            return data

    def preprocess(self):
        ''' Extracts data and converts each sample as an entry in a list '''
        table = self.get_data()
        table = table.split('\n')
        table = [cell.split(',') for cell in table]
        table = table[:-2]
        m = 0
        self.data = [{} for n in range(len(table))]
        for n in range(len(table)):
            m = 0
            sample = {}
            for feature in self.features[:-1]:
                sample[feature] = float(table[n][m])
                m += 1
            sample[self.features[-1]] = table[n][m]
            self.truth_vals.add(table[n][m])
            if table[n][m] in self.class_cnt:
                self.class_cnt[table[n][m]] += 1
            else:
                self.class_cnt[table[n][m]] = 1
            self.data[n] = sample

    def train(self):
        ''' Runs the clustering algorithm for the given number of iterations (10) '''
        means = np.random.choice(len(self.data), self.k, replace=False) # Random choice of means for the first iteration
        means = list(means)
        ans = []
        for n in range(self.k):
            means[n] = self.data[means[n]]

        for iter in range(self.iterations):
            cluster = []
            for n in range(len(self.data)):
                d = []
                for mean in means:
                    d.append(self.Euclidean(self.data[n], mean))    # Euclidean Distance of the sample from the currently chosen means
                cluster.append(np.argmin(d))
            ans = cluster
            new_means = []

            for n in range(self.k):
                sample = {}
                for feature in self.features[:-1]:
                    sample[feature] = 0
                new_means.append(sample)

            cluster_size = [0 for n in range(self.k)]

            for n in range(len(self.data)):
                group = cluster[n]
                cluster_size[group] += 1
                for feature in self.features[:-1]:
                    new_means[group][feature] += self.data[n][feature]

            for n in range(self.k):         # Updates means for the next iteration
                for feature in self.features[:-1]:
                    new_means[n][feature] /= cluster_size[n]
            means = new_means

        for n in range(self.k):
            for val in means[n]:
                means[n][val] = round(means[n][val], 3)

        return ans, means  # Returns the final cluster predicted for each of the sample

    def Euclidean(self, x, y):
        ''' Returns the Euclidean distance between two given data points '''
        distance = 0
        for feature in self.features[:-1]:
            distance += abs(x[feature] - y[feature]) ** 2
        return np.sqrt(distance)

    def Jaccard(self):
        ''' Computes the Jaccard Distance for each cluster from all the 3 possible class values 
                and predicts the class to be the one with the minimum Jaccard Distance '''
        groups = []
        ans = 0
        for n in range(self.k):
            all_possible_grps = {}
            for value in self.truth_vals:
                all_possible_grps[value] = 0
            groups.append(all_possible_grps)

        for n in range(len(self.data)):
            cluster = self.clusters[n]
            value = self.data[n][self.features[-1]]
            groups[cluster][value] += 1
        
        print("\n Clusters Formed")
        for n in range(self.k):
            print("   Cluster {} :".format(n), end=' ')
            print(groups[n])

        for n in range(self.k):
            min_dist = 10 ** 9 + 7
            value = ""
            total_samples = sum([groups[n][val] for val in groups[n]])

            print("\n Jaccard Distances for Cluster {}".format(n))
            print("\tCenter")
            print("\t    ",end=' ')
            print(self.means[n])
            print("\tFrom")

            for tr_val in self.truth_vals:
                print("\t   > ",end='')
                intersection = groups[n][tr_val]
                union = total_samples + self.class_cnt[tr_val] - intersection
                curr_dist = 1 - (intersection / union)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    value = tr_val
                print(tr_val + " = {}".format(curr_dist))
            print("\t+++ Predicted class for cluster {} is ".format(n) + "'" + value + "'" + " (Jaccard distance = {})".format(round(min_dist, 6)))
            ans += min_dist
        print("\n   Total Jaccard distance for the predicted classes = {}".format(round(ans, 6)))
        return

if __name__ == "__main__":
    K_means()