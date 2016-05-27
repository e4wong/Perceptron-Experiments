import sys
import random
import copy
from numpy import random as rn
from library import *
import matplotlib.pyplot as plt

def stepsize_fn(w, x, y):
    config = 1 
    if config == 1:
        return y

def update(w_t, x_t, y_t):
    w_update = None
    stepsize_policy = "constant"
    if stepsize_policy == "constant":
        w_update = [w + y_t * x for (w, x) in zip(w_t, x_t)]
    elif stepsize_policy == "margin":
        margin = dot_product(w_t, x_t)
        # KEY HERE print margin
        w_update = [w - margin * x for (w, x) in zip(w_t, x_t)]
    return w_update

def get_sample(error_samples, w, policy):
    if policy == "random":
        (features, label) = random.choice(error_samples)
        best_val = abs(dot_product(w, features))
        return ((features, label), best_val)
    elif policy == "large":
        best_val = abs(dot_product(w, error_samples[0][0]))
        best = error_samples[0]
        for (features, label) in error_samples:
            if abs(dot_product(w, features)) >= best_val:
                best_val = abs(dot_product(w, features))
                best = (features, label)
        return (best, best_val)
    elif policy == "small":
        best_val = abs(dot_product(w, error_samples[0][0]))
        best = error_samples[0]
        for (features, label) in error_samples:
            if abs(dot_product(w, features)) <= best_val:
                best_val = abs(dot_product(w, features))
                best = (features, label)
        return (best, best_val)
    else:
        print "ERROR! Unknown policy"

def perceptron(training_set, test_set, num_updates, policy):

    error_rate_trace = []
    margin_trace = []

    w = rn.normal(size=(1, len(training_set[0][0])))[0]
    counter = 0
    training_set = copy.deepcopy(training_set)

    error_rate_trace.append(calc_error_rate(w, test_set))

    while counter < num_updates:
        error_samples = [(features, label) for (features, label) in training_set if label * dot_product(w, features) <= 0]
        if len(error_samples) == 0:
            break

        ((features, label), margin) = get_sample(error_samples, w, policy)
        # remove chosen example
        training_set.remove((features, label))
        w = update(w, features, label)
        margin_trace.append(margin)
        error_rate_trace.append(calc_error_rate(w, test_set))
        counter += 1

    # to ensure all error_rate_trace is of length num_updates + 1...
    while len(error_rate_trace) != num_updates + 1:
        margin_trace.append(0.0)
        error_rate_trace.append(error_rate_trace[-1])

    return (w, error_rate_trace, margin_trace)

def active_learning(data, policy, max_updates, times_to_run):
    error_rate = []
    avg_error_rate_trace = []
    avg_margin_trace = []
    divide_by = times_to_run
    for i in range(0, times_to_run):
        random.shuffle(data)
        training_set = data[len(data)/2 : ]
        test_set = data[ : len(data)/2]
        (w, error_rate_trace, margin_trace) = perceptron(training_set, test_set, max_updates, policy)
        if len(avg_error_rate_trace) == 0:
            avg_error_rate_trace = error_rate_trace
        elif len(avg_error_rate_trace) == len(error_rate_trace):
            tmp = [avg_error_rate_trace[i] + error_rate_trace[i] for i in range(0, len(avg_error_rate_trace))]
            avg_error_rate_trace = tmp
        else:
            print "THIS SHOULD NOT HAPPEN PLEASE VISIT ME..."
            sys.exit(0)

        if len(avg_margin_trace) == 0:
            avg_margin_trace = margin_trace
        elif len(avg_margin_trace) == len(margin_trace):
            tmp = [avg_margin_trace[i] + margin_trace[i] for i in range(0, len(avg_margin_trace))]
            avg_margin_trace = tmp
        else:
            print "THIS SHOULD NOT HAPPEN PLEASE VISIT ME..."
            sys.exit(0)

        error_rate.append(calc_error_rate(w, test_set))
        if error_rate[-1] == 0.0:
            print w
    for i in range(0, len(avg_error_rate_trace)):
        avg_error_rate_trace[i] = avg_error_rate_trace[i]/divide_by
    for i in range(0, len(avg_margin_trace)):
        avg_margin_trace[i] = avg_margin_trace[i]/divide_by

    return (error_rate, avg_error_rate_trace, avg_margin_trace)
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
        default_policy = "random"

        (w_result, error_rate_trace, margin_trace) = perceptron(training_set, test_set, num_updates, default_policy)

        print "Error Rate of resulting w:", calc_error_rate(w_result, test_set)
        print "Error Rate of w*:", calc_error_rate(wstar, test_set)

        title = filename + default_policy
        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(title)
        plt.plot(error_rate_trace)
        plt.ylabel('Error Rate Trace')

        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(title)
        plt.plot(margin_trace)
        plt.ylabel("Abs(Margin)")

        output_final_w(w_result)
        plt.show()
    elif len(sys.argv) == 4:
        filename = sys.argv[1]
        num_updates = int(sys.argv[2])  
        num_runs = int (sys.argv[3])

        (wstar, data) = load_data(filename)
        random.shuffle(data)

        '''
        test_set = data[2*len(data)/3 : ]

        data = data[ : 2*len(data)/3]
        training_set = data[len(data)/2 : ]
        validation_set = data[ : len(data)/2]

        print "Length of Test Set:", len(test_set)
        print "Length of Validation Set:", len(validation_set)
        print "Length of Training Set:", (len(training_set))
        '''
        print "Running Random"
        (random_error_rate, random_err_trace, random_margin_trace) = active_learning(data, "random", num_updates, num_runs)
        print "Running Large"
        (large_error_rate, large_err_trace, large_margin_trace) = active_learning(data, "large", num_updates, num_runs)
        print "Running Small"
        (small_error_rate, small_err_trace, small_margin_trace) = active_learning(data, "small", num_updates, num_runs)
        print "Error Rate of w*:", calc_error_rate(wstar, data)

        print "Avg Error Rate Random:", avg(random_error_rate)
        print "Avg Error Rate Large:", avg(large_error_rate)
        print "Avg Error Rate Small:", avg(small_error_rate)
        
        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(filename)

        plt.plot(small_error_rate)
        plt.plot(large_error_rate)
        plt.plot(random_error_rate)
        plt.legend(["Small", "Large", "Random"], loc='upper left')
        plt.ylabel('Error Rate')

        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(filename)
        plt.plot(small_err_trace)
        plt.plot(large_err_trace)
        plt.plot(random_err_trace)
        plt.legend(["Small", "Large", "Random"], loc='upper right')
        plt.ylabel("Error Rate Trace")

        f_2 = plt.figure(2)
        f_2.canvas.set_window_title(filename)
        plt.plot(small_margin_trace)
        plt.plot(large_margin_trace)
        plt.plot(random_margin_trace)
        plt.legend(["Small", "Large", "Random"], loc='upper right')
        plt.ylabel("Abs(Margin) Trace")

        plt.show()


if __name__ == "__main__":
    main() 
