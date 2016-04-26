import sys
import random
from numpy import random as rn
from library import *

def stepsize_fn(w, x, y):
    config = 1 
    if config == 1:
        return y



def update(w_t, x_t, y_t):
    w_update = [ w + y_t * x for (w, x) in zip(w_t, x_t)]
    return w_update

def perceptron(training_set, test_set, max_updates = None):
    if max_updates is None:
        max_updates = len(training_set)

    w = rn.normal(size=(1, len(training_set[0][0])))[0]
    
    counter = 0
    for (features, label) in training_set:
        if counter >= max_updates:
            break

        if label * dot_product(w, features) <= 0:
            w = update(w, features, label)
            counter += 1

    print "Number of updates:", counter
    return w
# Parameters:
# 1. Dataset
# 2. Stepsize? constant or margin? maybe just configurable in code for now
# 3. number of updates to do/passes to do in training stage
def main():
    if len(sys.argv) < 3:
        print "Wrong # of args"
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        num_updates = int(sys.argv[2])     
        (wstar, data) = load_data(filename)
        random.shuffle(data)

        test_set = data[2*len(data)/3 : ]

        data = data[ : 2*len(data)/3]
        training_set = data[len(data)/2 : ]
        validation_set = data[ : len(data)/2]

        print "Length of Test Set:", len(test_set)
        print "Length of Validation Set:", len(validation_set)
        print "Length of Training Set:", (len(training_set))

        w_result = perceptron(training_set, test_set, 10)
        
        print "Error Rate of resulting w:", calc_error_rate(w_result, test_set)
        print "Error Rate of w*:", calc_error_rate(wstar, test_set)



if __name__ == "__main__":
    main() 
