from collections import defaultdict
from nltk.util import ngrams
from nltk.probability import FreqDist, WittenBellProbDist
from math import log, exp
from sys import float_info


min_log_prob = -float_info.max

# function to calculate the log sum of exponentials


def logsumexp(vals):
    if len(vals) == 0:
        return min_log_prob
    m = max(vals)
    if m == min_log_prob:
        return min_log_prob
    else:
        return m + log(sum([exp(val - m) for val in vals]))


# Count occurrences of one part of speech following another
def get_transition_counts(sents):
    transition_counts = defaultdict(FreqDist)
    for sent in sents:
        tags = ['<s>'] + [token['upos'] for token in sent] + ['</s>']
        for tag1, tag2 in ngrams(tags, 2):
            transition_counts[tag1][tag2] += 1

    return transition_counts

# Count occurrences of words together with parts of speech


def get_emission_counts(sents):
    emission_counts = defaultdict(FreqDist)
    for sent in sents:
        for token in sent:
            word = token['form']
            tag = token['upos']
            emission_counts[tag][word] += 1

    return emission_counts

# Estimate transition probabilities with Witten-Bell smoothing


def get_transition_probs(transition_counts):
    # Create a probability distribution for each tag
    transition_probs = defaultdict(WittenBellProbDist)
    for tag in transition_counts:
        # Create a frequency distribution for the tags that follow the tag
        freq_dist = transition_counts[tag]
        transition_probs[tag] = WittenBellProbDist(freq_dist, bins=1e5)

    return transition_probs

# Estimate emission probabilities with Witten-Bell smoothing


def get_emission_probs(emission_counts):
    # Create a probability distribution for each tag
    emission_probs = defaultdict(WittenBellProbDist)
    for tag in emission_counts:
        # Create a frequency distribution for the words associated with the tag
        freq_dist = emission_counts[tag]
        emission_probs[tag] = WittenBellProbDist(freq_dist, bins=1e5)

    return emission_probs


# Fucntion to print the probabilities
def print_probs(transition_probs, emission_probs):
    print('Transition probabilities:')
    for tag1 in transition_probs:
        for tag2 in transition_probs[tag1].samples():
            print(tag1, '->', tag2, ':', transition_probs[tag1].prob(tag2))
    print()
    print('Emission probabilities:')
    for tag in emission_probs:
        for word in emission_probs[tag].samples():
            print(tag, '->', word, ':', emission_probs[tag].prob(word))

# function to calculate the accuracy


def calculate_accuracy(pred, actual):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            correct += 1

    return correct / len(pred)

# function to calculate the frequency accuracy


def calculate_freq_accuracy(predicted_tags, actual_tags):
    freq_accuracy = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    for i in range(len(predicted_tags)):
        if predicted_tags[i] == actual_tags[i]:
            freq_accuracy[actual_tags[i]]['correct'] += 1
        else:
            freq_accuracy[actual_tags[i]]['incorrect'] += 1

    return freq_accuracy
