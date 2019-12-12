# 17EC10063
# Kshitij Agrawal
# Assignment 2 (Naïve Bayes Classifier)

# To run : python3 17EC10063_2.py
# (Won't work with python2, so please ensure to explicitly mention python3)
# To test or train on another dataset, change the name of the train_file and(or) test_file to the new file(s)

train_file = "data2_19.csv"
test_file = "test2_19.csv"

class Bayes_classifer:
    ''' Naïve Bayes Classifier class, houses the functions to process the dataset
            and make predictions from it '''
    def __init__(self):
        self.train, self.labels = self.get_data(train_file)
        self.test, self.labels = self.get_data(test_file)
        self.target = {}
        self.features = {}
        self.classes = []
        self.preprocess()              # Calculates all the required probabilities
        self.predict(self.train, 0)    # Accuracy on training set
        self.predict(self.test)        # Accuracy on test set

    def get_data(self, file):
        with open(file, 'r') as f:
            data = f.read()
            return self.extract_data(data)

    def extract_data(self, data):
        data = data.split('\n')
        data = data[:-1]
        data = [val[1:-1] for val in data]  #  Remove unwanted string endings ("") 
        data = [vals.split(',') for vals in data]
        self.labels = data[0]
        data = data[1:]
        return data, self.labels

    def get_result_index(self):
        ''' Returns the position of the output class in the list of labels'''
        result = 'D'
        pos = 0
        for label in self.labels:
            if label == result:
                break
            pos += 1
        return pos

    def preprocess(self):
        self.target = {}
        result = 'D'
        result_index = self.get_result_index()
        possible_results = set()
        for values in self.train:   #  Calculates the prior probability of each outcome class
            outcome = values[result_index]
            possible_results.add(outcome)
            if outcome in self.target:
                self.target[outcome] += 1
            else:
                self.target[outcome] = 1
        pos = 0
        self.classes = list(possible_results)   #   All the possible outcomes (0 and 1 in our case)

        for label in self.labels:
            if label is not result:
                self.features[label] = {}
                for results in possible_results:
                    self.features[label][results] =  {}
                for values in self.train:       #   Calculates the probability P(X_i = x_ij | Y = y_k)
                                                #   x_ij takes values in [1, 5], X_i takes values [X1, X6] and y_k {0, 1}
                    outcome = values[result_index]
                    rating = values[pos]
                    if rating in self.features[label][outcome]:
                        self.features[label][outcome][rating] += 1
                    else:
                        self.features[label][outcome][rating] = 1
            pos += 1
        return

    def laplace_smoothing(self, a, b):   
        ''' Returns the probabily P(A | B) after applying Laplace Smoothing '''
        ''' a is the number of samples in the given feature with result class B, having b samples
                and n ( = 5) classes in the given feature '''
        return (a + 1) / (b + 5)

    def predict(self, dataset, flag = 1):
        ''' Predicts the output on the given dataset and Calculates the accuarcy of the build classifier '''
        result_index = self.get_result_index()
        correct_predict = 0

        for values in dataset:
            expected_result = values[result_index]
            actual_result = 0
            total_outcomes = len(self.train)
            answer = 0  #   Stores the max probability obtained on choosing one output class
            for current_class in self.classes:
                product = self.laplace_smoothing(self.target[current_class], total_outcomes)
                pos = 0
                for vals in values:
                    if pos is not result_index:
                        new_rating = vals   #   Given value of x_ij in the test set for the feature X_i
                        label = self.labels[pos]
                        den = self.target[current_class]
                        num = 0
                        if new_rating in self.features[label][current_class]:
                            num = self.features[label][current_class][new_rating]
                        else:
                            num = 0
                        product *= self.laplace_smoothing(num, den)
                    pos += 1
                if product > answer:    #   Update the answer (max probability so far) and hence the predicted class for given data
                    answer = product
                    actual_result = current_class

            correct_predict += (expected_result == actual_result)   #   Count of correctly predicted samples
        accuracy = correct_predict / len(dataset)
        accuracy *= 100
        accuracy = round(accuracy, 2)
        file = None
        if flag:
            file = "Test "
        else:
            file = "Train"
        print("Accuracy on {} set =".format(file), end = ' ')
        print(str(accuracy) + " %")
        return

if __name__ == "__main__":
    Classifier = Bayes_classifer()    