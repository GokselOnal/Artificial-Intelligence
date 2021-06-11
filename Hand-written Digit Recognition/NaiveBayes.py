from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

class NBY:

    def __init__(self, train_percent=0.75):
        # self.digits = fetch_openml('mnist_784')
        self.digits = load_digits()
        self.train_percent = train_percent
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.digits.data, self.digits.target, train_size= self.train_percent,random_state=0)

    def train(self,classifier):
        if (classifier == "Gaussian"):
            nby_classifier = GaussianNB()
        elif (classifier == "Multinomial"):
            nby_classifier = MultinomialNB()
        elif (classifier == "Complement"):
            nby_classifier = ComplementNB()
        elif (classifier == "Bernoulli"):
            nby_classifier = BernoulliNB()

        nby_fit = nby_classifier.fit(self.x_train, self.y_train)
        nby_prediction = nby_fit.predict(self.x_test)

        return accuracy_score(self.y_test, nby_prediction)

    def validate(self): #for proof purposes
        accuracy_list_gaussian = list()
        accuracy_list_multinomial = list()
        accuracy_list_complement = list()
        accuracy_list_bernoulli = list()

        accuracy_list_gaussian.append(self.train("Gaussian"))
        accuracy_list_multinomial.append(self.train("Multinomial"))
        accuracy_list_complement.append(self.train("Complement"))
        accuracy_list_bernoulli.append(self.train("Bernoulli"))

        plt.title("NBY(classifier)")
        plt.plot(accuracy_list_gaussian, color="r", marker = "*", label="Gaussian")
        plt.plot(accuracy_list_multinomial, color="b", marker="*", label="Multinomial")
        plt.plot(accuracy_list_complement, color ="y",marker="*", label="Complement")
        plt.plot(accuracy_list_bernoulli, color = "g",marker="*", label="Bernoulli")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        accuracy_list = list()
        accuracy_list.append(accuracy_list_gaussian)
        accuracy_list.append(accuracy_list_multinomial)
        accuracy_list.append(accuracy_list_complement)
        accuracy_list.append(accuracy_list_bernoulli)

        if (accuracy_list.index(max(accuracy_list)) == 0):
            print("Selected classifier => Gaussian")
            return "Gaussian"
        elif (accuracy_list.index(max(accuracy_list)) == 1):
            print("Selected classifier => Multinomial")
            return "Multinomial"
        elif (accuracy_list.index(max(accuracy_list)) == 2):
            print("Selected classifier => Complement")
            return "Complement"
        elif (accuracy_list.index(max(accuracy_list)) == 3):
            print("Selected classifier => Bernoulli")
            return "Bernoulli"

    def train_multinomial(self, alpha):
        nby_classifier_mul = MultinomialNB(alpha=alpha)
        nby_fit_mul = nby_classifier_mul.fit(self.x_train, self.y_train)
        nby_prediction_mul = nby_fit_mul.predict(self.x_test)
        return accuracy_score(self.y_test, nby_prediction_mul)

    def validate2(self):
        alpha = 0.01
        accuracy_list = list()
        x_plot = list()
        for i in range(99):
            x_plot.append(alpha)
            accuracy_list.append(self.train_multinomial(alpha= alpha))
            alpha += 0.010
        plt.title("NBY(alpha)")
        plt.plot(x_plot, accuracy_list)
        plt.xlabel("alpha")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
        return (accuracy_list.index(max(accuracy_list)) + 1) /100

    def result(self):
        print("NBY")
        self.validate() #proof
        optimal_alpha = self.validate2()
        print("asdasd",optimal_alpha)
        nby_classifier = MultinomialNB(alpha=optimal_alpha)
        nby_fit = nby_classifier.fit(self.x_train, self.y_train)
        nby_prediction = nby_fit.predict(self.x_test)

        def confusion_matrix_train():
            nby_classifier_train = MultinomialNB()
            nby_fit_train = nby_classifier_train.fit(self.x_train, self.y_train)
            matrix = plot_confusion_matrix(nby_classifier_train, self.x_train, self.y_train, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Train Confusion Matrix(NBY)")
            print(f"Train Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_train():
            nby_classifier_train = MultinomialNB()
            nby_fit_train = nby_classifier_train.fit(self.x_train, self.y_train)
            nby_prediction_train = nby_fit_train.predict(self.x_train)
            print(f"Classification report for train classifier {nby_classifier_train}:\n"
            f"{metrics.classification_report(self.y_train, nby_prediction_train)}\n")
            print("Accuracy:", accuracy_score(self.y_train, nby_prediction_train),"\n")

        def confusion_matrix_test(): #with tuned params
            matrix = plot_confusion_matrix(nby_classifier,self.x_test,self.y_test, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Test Confusion Matrix(NBY)")
            print(f"Confusion matrix:\n{matrix.confusion_matrix}\n")
            plt.show()

        def report_test():
            print(f"Classification report for test classifier {nby_classifier}:\n"
            f"{metrics.classification_report(self.y_test, nby_prediction)}\n")
            print("Accuracy:", accuracy_score(self.y_test, nby_prediction),"\n")

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