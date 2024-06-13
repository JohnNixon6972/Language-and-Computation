from treebanks import conllu_corpus, train_corpus, test_corpus
from helper_functions import get_transition_counts, get_emission_counts, get_transition_probs, get_emission_probs, calculate_accuracy, calculate_freq_accuracy
from eager_algo import eager_algorithm

from viterbi_algo import viterbi_algorithm
from baum_welch_algo import individually_most_probable_tags
from eager_algo import eager_algorithm
import matplotlib.pyplot as plt
import numpy as np

def main():

    langs = ['en', 'sv', 'ko', 'ja']
    freq_graphs = []

    for lang in langs:
        print('*' * 60)
        # Load the training and test corpora
        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))

        print('language:', lang)
        print(len(train_sents), 'training sentences')
        print(len(test_sents), 'test sentences')

        # Get the transition and emission counts
        transition_counts = get_transition_counts(train_sents)
        emission_counts = get_emission_counts(train_sents)

        # Estimate the transition and emission probabilities
        transition_probs = get_transition_probs(transition_counts)
        emission_probs = get_emission_probs(emission_counts)

        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0

        predicted_tags1_total = []
        predicted_tags2_total = []
        predicted_tags3_total = []
        actual_tags_total = []

        # Iterate over the test sentences
        for sent in test_sents:
            # get the actual tags
            actual_tags = ['<s>'] + [token['upos'] for token in sent] + ['</s>']

            # get the predicted tags
            predicted_tags1 = eager_algorithm(sent, transition_probs, emission_probs)
            predicted_tags2 = viterbi_algorithm(sent, transition_probs, emission_probs)
            predicted_tags3 = individually_most_probable_tags(sent, transition_probs, emission_probs)

            # calculate the accuracy
            accuracy1 += calculate_accuracy(predicted_tags1, actual_tags)
            accuracy2 += calculate_accuracy(predicted_tags2, actual_tags)
            accuracy3 += calculate_accuracy(predicted_tags3, actual_tags)

            # add the predicted and actual tags to the total lists
            predicted_tags1_total += predicted_tags1
            predicted_tags2_total += predicted_tags2
            predicted_tags3_total += predicted_tags3
            actual_tags_total += actual_tags

        n = len(test_sents)
        print()
        print('*' * 30)
        print('Accuracy of Eager Algorithm:', ((accuracy1 / n) * 100), '%')
        print('Accuracy Viterbi Algorithm:', ((accuracy2 / n) * 100), '%')
        print('Accuracy Individually Most Probable (Baum-Welch Algorithm):', ((accuracy3 / n) * 100), '%')

        # Calculate frequency accuracy and append to freq_graphs
        freq_accuracy1 = calculate_freq_accuracy(predicted_tags1_total, actual_tags_total)
        freq_accuracy2 = calculate_freq_accuracy(predicted_tags2_total, actual_tags_total)
        freq_accuracy3 = calculate_freq_accuracy(predicted_tags3_total, actual_tags_total)

        freq_graphs.append((lang, freq_accuracy1, freq_accuracy2, freq_accuracy3))

    # Plot frequency accuracy graphs for each language
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Frequency Accuracy')

    for i, (lang, freq_acc1, freq_acc2, freq_acc3) in enumerate(freq_graphs):
        ax = axs[i // 2, i % 2]
        keys = list(freq_acc1.keys())
        indices = np.arange(len(keys))
        bar_width = 0.25
        
        correct1 = [d['correct'] for d in freq_acc1.values()]
        incorrect1 = [d['incorrect'] for d in freq_acc1.values()]
        correct2 = [d['correct'] for d in freq_acc2.values()]
        incorrect2 = [d['incorrect'] for d in freq_acc2.values()]
        correct3 = [d['correct'] for d in freq_acc3.values()]
        incorrect3 = [d['incorrect'] for d in freq_acc3.values()]
        
        ax.barh(indices - bar_width, correct1, bar_width, label='Eager Algorithm - Correct', alpha=0.7)
        ax.barh(indices - bar_width, incorrect1, bar_width, label='Eager Algorithm - Incorrect', alpha=0.7)
        ax.barh(indices, correct2, bar_width, label='Viterbi Algorithm - Correct', alpha=0.7)
        ax.barh(indices, incorrect2, bar_width, label='Viterbi Algorithm - Incorrect', alpha=0.7)
        ax.barh(indices + bar_width, correct3, bar_width, label='Individually Most Probable - Correct', alpha=0.7)
        ax.barh(indices + bar_width, incorrect3, bar_width, label='Individually Most Probable - Incorrect', alpha=0.7)
        
        ax.set_yticks(indices)
        ax.set_yticklabels(keys)
        ax.set_title(f'Language: {lang}')
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
