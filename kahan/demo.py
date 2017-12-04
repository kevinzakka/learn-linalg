import numpy as np

from sum import kahan_sum


def main():

    # create pathological input
    x = np.random.uniform(0, 1, int(1e8))



    naive_sum = 0.
    for i in range(len(x)):
        naive_sum += x[i]

    numpy_sum = x.sum()
    our_sum = kahan_sum(x)

    print("Numpy: {}".format(numpy_sum))
    print("Naive: {}".format(naive_sum))
    print("Kahan: {}".format(our_sum))
    print("Diff (Naive-Kahan): {}".format(np.abs(naive_sum-our_sum)))
    print("Diff (Numpy-Kahan): {}".format(np.abs(numpy_sum-our_sum)))


if __name__ == '__main__':
    main()
