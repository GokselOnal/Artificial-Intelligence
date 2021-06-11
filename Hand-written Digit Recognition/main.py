from KNearestNeighbor import KNN
from NaiveBayes import NBY
from DecisionTree import DCT
from time import time

while True:
    print("\n----------Hand-written Digit Recognition----------\n"
          "1. KNearestNeighbor(KNN)\n"
          "2. NaïveBayes(NBY)\n"
          "3. DecisionTree(DCT)\n"
          "4. Exit\n")

    print("Enter an algorithm:")
    option = int(input())
    algorithm = None

    if option == 1:
        algorithm = KNN()
    elif option == 2:
        algorithm = NBY()
    elif option == 3:
        algorithm = DCT()
    elif option == 4:
        print("Exiting...")
        break
    else:
        raise AssertionError("Invalid input for algorithm selection!")

    init_time = time()
    algorithm.result()
    elapsed_time = time() - init_time
    print("Elapsed time: {} secs.".format(elapsed_time))