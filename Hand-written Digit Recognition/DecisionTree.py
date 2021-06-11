from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

class DCT:

    def __init__(self, train_percent = 0.75):
        # self.digits = fetch_openml('mnist_784')
        self.digits = load_digits()
        self.train_percent = train_percent
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.digits.data, self.digits.target, train_size= self.train_percent, random_state=0)


    def train(self,criterion = "gini",splitter="best"):
        if(criterion == "gini" and splitter == "best"):
            dct_classifier = tree.DecisionTreeClassifier(criterion="gini", splitter="best")
        elif(criterion== "gini" and splitter == "random"):
            dct_classifier = tree.DecisionTreeClassifier(criterion="gini", splitter="random")
        elif(criterion== "entropy" and splitter == "best"):
            dct_classifier = tree.DecisionTreeClassifier(criterion="entropy", splitter="best")
        elif (criterion == "entropy" and splitter == "random"):
            dct_classifier = tree.DecisionTreeClassifier(criterion="entropy", splitter="random")

        dct_fit = dct_classifier.fit(self.x_train, self.y_train)
        dct_prediction = dct_fit.predict(self.x_test)
        return accuracy_score(self.y_test, dct_prediction)

    def validate(self):
        accuracy_gini_best = list()
        accuracy_gini_random = list()
        accuracy_entropy_best = list()
        accuracy_entropy_random = list()

        for i in range(15):
            accuracy_gini_best.append(self.train(criterion="gini", splitter="best"))
            accuracy_gini_random.append(self.train(criterion="gini", splitter="random"))
            accuracy_entropy_best.append(self.train(criterion="entropy", splitter="best"))
            accuracy_entropy_random.append(self.train(criterion="entropy", splitter="random"))

        plt.title("DCT(criterion,splitter)")
        plt.plot(accuracy_gini_best, color="r", marker="o", label="gini,best")
        plt.plot(accuracy_gini_random, color="b", marker="o", label="gini,random")
        plt.plot(accuracy_entropy_best, color="y", marker="o", label="entropy,best")
        plt.plot(accuracy_entropy_random, color="g", marker="o", label="entropy,random")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        accuracy_list = list()

        sum = 0
        for i in accuracy_gini_best:
            sum += i
        accuracy_list.append(sum/15)

        sum = 0
        for i in accuracy_gini_random:
            sum += i
        accuracy_list.append(sum/15)

        sum = 0
        for i in accuracy_entropy_best:
            sum += i
        accuracy_list.append(sum /15)

        sum = 0
        for i in accuracy_entropy_random:
            sum += i
        accuracy_list.append(sum / 15)

        if(accuracy_list.index(max(accuracy_list)) == 0):
            print("Selected criterion,splitter => gini,best")
            return "gini best"
        elif (accuracy_list.index(max(accuracy_list)) == 1):
            print("Selected criterion,splitter => gini,random")
            return "gini random"
        elif (accuracy_list.index(max(accuracy_list)) == 2):
            print("Selected criterion,splitter => entropy,best")
            return "entropy best"
        elif (accuracy_list.index(max(accuracy_list)) == 3):
            print("Selected criterion,splitter => entropy,random")
            return "entropy random"


    def result(self):
        print("DCT")
        optimal_parameter = self.validate()
        parameters = optimal_parameter.split(" ")
        dct_classifier = tree.DecisionTreeClassifier(criterion=parameters[0], splitter=parameters[1])
        dct_fit = dct_classifier.fit(self.x_train, self.y_train)
        dct_prediction = dct_fit.predict(self.x_test)

        def confusion_matrix_train():
            dct_classifier_train = tree.DecisionTreeClassifier()
            dct_fit_train = dct_classifier_train.fit(self.x_train, self.y_train)
            matrix = plot_confusion_matrix(dct_classifier_train, self.x_train, self.y_train, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Train Confusion Matrix(DCT)")
            print(f"Train Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_train():
            dct_classifier_train = tree.DecisionTreeClassifier()
            dct_fit_train = dct_classifier_train.fit(self.x_train, self.y_train)
            dct_prediction_train = dct_fit_train.predict(self.x_train)
            print(f"Classification report for train classifier {dct_classifier_train}:\n"
            f"{metrics.classification_report(self.y_train, dct_prediction_train)}\n")
            print("Accuracy:", accuracy_score(self.y_train, dct_prediction_train), "\n")

        def confusion_matrix_test():
            matrix = plot_confusion_matrix(dct_classifier,self.x_test,self.y_test, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Test Confusion Matrix(DCT)")
            print(f"Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_test():
            print(f"Classification report for test classifier {dct_classifier}:\n"
            f"{metrics.classification_report(self.y_test, dct_prediction)}\n")
            print("Accuracy:", accuracy_score(self.y_test, dct_prediction), "\n")

        confusion_matrix_train()
        report_train()
        confusion_matrix_test()
        report_test()

    def visualize_images(self):
        _, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
        for ax, image, label in zip(axes, self.digits.images, self.digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Digit: %i' % label)
        plt.show()
