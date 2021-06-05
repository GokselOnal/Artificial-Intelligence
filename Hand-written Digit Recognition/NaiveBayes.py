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

    def train(self,classifier):
        x_train, x_test, y_train, y_test = train_test_split(self.digits.data, self.digits.target, train_size =self.train_percent)

        if (classifier == "Gaussian"):
            nby_classifier = GaussianNB()
        elif (classifier == "Multinomial"):
            nby_classifier = MultinomialNB()
        elif (classifier == "Complement"):
            nby_classifier = ComplementNB()
        elif (classifier == "Bernoulli"):
            nby_classifier = BernoulliNB()

        nby_fit = nby_classifier.fit(x_train, y_train)
        nby_prediction = nby_fit.predict(x_test)

        return accuracy_score(y_test, nby_prediction)

    def validate(self):
        accuracy_list_gaussian = list()
        accuracy_list_multinomial = list()
        accuracy_list_complement = list()
        accuracy_list_bernoulli = list()

        for i in range(5):
            accuracy_list_gaussian.append(self.train("Gaussian"))
            accuracy_list_multinomial.append(self.train("Multinomial"))
            accuracy_list_complement.append(self.train("Complement"))
            accuracy_list_bernoulli.append(self.train("Bernoulli"))

        accuracy_list = list()
        sum = 0
        for i in accuracy_list_gaussian:
            sum += i
        accuracy_list.append(sum/len(accuracy_list_gaussian))

        sum = 0
        for i in accuracy_list_multinomial:
            sum += i
        accuracy_list.append(sum / len(accuracy_list_gaussian))

        sum = 0
        for i in accuracy_list_complement:
            sum += i
        accuracy_list.append(sum / len(accuracy_list_gaussian))

        sum = 0
        for i in accuracy_list_bernoulli:
            sum += i
        accuracy_list.append(sum / len(accuracy_list_gaussian))

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

    def train_multinomial(self, alpha= 1.0):
        x_train, x_test, y_train, y_test = train_test_split(self.digits.data, self.digits.target, train_size = self.train_percent)
        nby_classifier = MultinomialNB(alpha=alpha)
        nby_fit = nby_classifier.fit(x_train, y_train)
        nby_prediction = nby_fit.predict(x_test)
        return accuracy_score(y_test, nby_prediction)

    """def validate2(self):
        alpha = 0.01
        accuracy_list = list()
        for i in range(90):
            accuracy_list.append(self.train_multinomial(alpha= alpha))
            alpha += 0.010
        print("alpha optimal alpha = >",max(accuracy_list),(accuracy_list.index(max(accuracy_list)) + 1)/100)"""


    def result(self):
        x_train, x_test, y_train, y_test = train_test_split(self.digits.data, self.digits.target, train_size=self.train_percent)
        optimal_cls = self.validate() #for finding optimal cls
        if (optimal_cls == "Gaussian"):
            nby_classifier = GaussianNB()
        elif (optimal_cls == "Multinomial"):
            nby_classifier = MultinomialNB()
        elif (optimal_cls == "Complement"):
            nby_classifier = ComplementNB()
        elif (optimal_cls == "Bernoulli"):
            nby_classifier = BernoulliNB()
        nby_fit = nby_classifier.fit(x_train, y_train)
        nby_prediction = nby_fit.predict(x_test)

        def confusion_matrix():
            matrix = plot_confusion_matrix(nby_classifier,x_test,y_test, cmap=plt.cm.viridis)
            matrix.figure_.suptitle("Confusion Matrix(NBY)")
            print(f"NBY\nConfusion matrix:\n{matrix.confusion_matrix}")
            plt.show()

        def report(): # 4 tanesi
            print(f"Classification report for classifier {nby_classifier}:\n"
            f"{metrics.classification_report(y_test, nby_prediction)}\n")

        confusion_matrix()
        report()

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