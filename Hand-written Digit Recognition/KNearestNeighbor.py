from sklearn.datasets import *
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

class KNN:
    def __init__(self, train_percent = 0.75):
        # self.digits = fetch_openml('mnist_784')
        self.digits = load_digits()
        self.train_percent = train_percent

    def train(self, neighbors= 5):
        x_train, x_test, y_train, y_test = train_test_split(self.digits.data, self.digits.target, train_size= self.train_percent)
        knn_classifier = KNeighborsClassifier(n_neighbors= neighbors)
        knn_fit = knn_classifier.fit(x_train, y_train)
        knn_prediction = knn_fit.predict(x_test)
        return accuracy_score(y_test, knn_prediction)

    def validate(self):
        neighbors = 1
        accuracy_list = list()
        for i in range(1, 42):
            if (i ** 2 < 0.75 * len(self.digits.data)):
                accuracy_list.append(self.train(neighbors= neighbors))
                neighbors = i ** 2
        k = ((accuracy_list.index(max(accuracy_list))) + 1) ** 2
        print("Max Accuracy", max(accuracy_list), "\nK =>", k)
        return k

    def validate2(self):
        neighbors = 1
        accuracy_list = list()
        for i in range(10):
            accuracy_list.append(self.train(neighbors=neighbors))
            neighbors += 1
        k = (accuracy_list.index(max(accuracy_list))) + 1
        print("Selected K =>", k)
        return k


    def result(self):
        optimal_n = self.validate2()
        x_train, x_test, y_train, y_test = train_test_split(self.digits.data, self.digits.target,train_size=  self.train_percent)
        knn_classifier = KNeighborsClassifier(n_neighbors= optimal_n)
        knn_fit = knn_classifier.fit(x_train, y_train)
        knn_prediction = knn_fit.predict(x_test)
        def confusion_matrix():
            matrix = plot_confusion_matrix(knn_classifier, x_test, y_test, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Confusion Matrix(KNN)")
            print(f"KNN\nConfusion matrix:\n{matrix.confusion_matrix}")
            plt.show()
        def report():
            print(f"Classification report for classifier {knn_classifier}:\n"
            f"{metrics.classification_report(y_test, knn_prediction)}\n")

        report()
        confusion_matrix()


    def display_image(self, i):
        plt.gray()
        plt.matshow(self.digits.images[i])
        plt.show()

    def visualize_images(self):
        _, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
        for ax, image, label in zip(axes, self.digits.images, self.digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Digit: %i' % label)
            plt.show()
