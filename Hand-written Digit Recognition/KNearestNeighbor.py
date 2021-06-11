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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.digits.data, self.digits.target, train_size= self.train_percent, random_state=0)


    def train(self, neighbors= 5):
        knn_classifier = KNeighborsClassifier(n_neighbors= neighbors)
        knn_fit = knn_classifier.fit(self.x_train, self.y_train)
        knn_prediction = knn_fit.predict(self.x_test)
        return accuracy_score(self.y_test, knn_prediction)

    def validate(self): #for proof purposes
        neighbors = 1
        accuracy_list = list()
        x_plot = list()
        for i in range(2, 42):
            if (i ** 2 < self.train_percent * len(self.digits.data)):
                x_plot.append(neighbors)
                accuracy_list.append(self.train(neighbors= neighbors))
                neighbors = i ** 2
        k = ((accuracy_list.index(max(accuracy_list))) + 1) ** 2
        print("Max Accuracy", max(accuracy_list), "\nK =>", k)
        plt.scatter(x_plot,accuracy_list)
        plt.title("KNN(n_neighbors)")
        plt.xlabel("Neighbor")
        plt.ylabel("Accuracy")
        plt.show()
        return k

    def validate2(self):
        accuracy_list = list()
        x_plot = list()
        for i in range(1,10):
            x_plot.append(i)
            accuracy_list.append(self.train(neighbors=i))
            i += 1
        k = (accuracy_list.index(max(accuracy_list))) + 1
        plt.title("KNN(n_neighbors)")
        plt.plot(x_plot, accuracy_list)
        plt.xlabel("Neighbor")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
        return k

    def result(self):
        print("KNN")
        self.validate() #proof
        optimal_n = self.validate2()
        knn_classifier = KNeighborsClassifier(n_neighbors= optimal_n)
        knn_fit = knn_classifier.fit(self.x_train, self.y_train)
        knn_prediction = knn_fit.predict(self.x_test)

        def confusion_matrix_train(): #with tuned params
            knn_classifier_train = KNeighborsClassifier()
            knn_fit_train = knn_classifier_train.fit(self.x_train, self.y_train)
            matrix = plot_confusion_matrix(knn_classifier_train, self.x_train, self.y_train, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Train Confusion Matrix(KNN)")
            print(f"Train Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_train():
            knn_classifier_train = KNeighborsClassifier()
            knn_fit_train = knn_classifier_train.fit(self.x_train, self.y_train)
            knn_prediction_train = knn_fit_train.predict(self.x_train)
            print(f"Train Classification report for classifier {knn_classifier_train}:\n"
            f"{metrics.classification_report(self.y_train, knn_prediction_train)}\n")
            print("Accuracy:", accuracy_score(self.y_train, knn_prediction_train),"\n")

        def confusion_matrix_test():
            matrix = plot_confusion_matrix(knn_classifier, self.x_test, self.y_test, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Test Confusion Matrix(KNN)")
            print(f"Test Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_test():
            print("Optimal n_neighbors =>", optimal_n)
            print(f"Test Classification report for classifier {knn_classifier}:\n"
            f"{metrics.classification_report(self.y_test, knn_prediction)}\n")
            print("Accuracy:", accuracy_score(self.y_test, knn_prediction),"\n")

        confusion_matrix_train()
        report_train()
        confusion_matrix_test()
        report_test()

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
