# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    correct_num ,total_num = 0, 0
    for line in open('birth_dev.tsv',encoding='utf-8'):
        x = line.split('\t')[1]
        total_num += 1
        if x[:6] == 'London':
            correct_num += 1
    accuracy = correct_num/total_num*100
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
