from helper_functions import min_log_prob
from math import log, exp

# Viterbi algorithmc


def viterbi_algorithm(sent, transition_probs, emission_probs):
    strart_tag = '<s>'
    predicted_tags = []
    end_tag = '</s>'

    # Create a list of dictionaries to store the probabilities and the backpointers
    probs = [{}]
    backpointers = [{}]

    # Initialize the probability of the start tag as 1
    probs[0][strart_tag] = 1

    # Iterate over the tokens in the sentence
    for token in sent:
        word = token['form']

        # Create a new dictionary for the current token
        probs.append({})
        backpointers.append({})

        # Iterate over the possible tags for the current token
        for tag in transition_probs:
            max_prob = min_log_prob
            max_tag = None

            # Iterate over the possible tags for the previous token
            for prev in transition_probs:
                # Calculate the probability using both transition and emission probabilities

                # If the word is not in the emission probabilities, assign a probability of 0
                ems_prob = min_log_prob
                if emission_probs.get(tag):
                    ems_prob = log(emission_probs.get(tag).prob(word))

                trans_prob = min_log_prob
                # If the tag is not in the transition probabilities, assign a probability of 0
                if transition_probs.get(prev):
                    trans_prob = log(transition_probs.get(prev).prob(tag))

                mul_prob = exp(ems_prob + trans_prob)

                # Calculate the probability of the previous tag
                if probs[-2].get(prev):
                    prob = probs[-2].get(prev) * mul_prob
                else:
                    prob = min_log_prob

                # Update the maximum probability and the tag with the maximum probability
                if prob > max_prob:
                    max_prob = prob
                    max_tag = prev

            # Store the maximum probability and the backpointer
            probs[-1][tag] = max_prob
            backpointers[-1][tag] = max_tag
            # print(max_prob, max_tag)

    # Add the tag with the maximum probability to the list of predicted tags
    max_tag = max(probs[-1], key=probs[-1].get)
    predicted_tags.append(max_tag)

    # Backtrack to find the best sequence of tags
    for i in range(len(probs) - 1, 0, -1):
        # print(backpointers[i].get(max_tag))
        max_tag = backpointers[i].get(max_tag)
        predicted_tags.append(max_tag)

    # Reverse the list of predicted tags
    predicted_tags.reverse()

    # Add the end tag
    predicted_tags.append(end_tag)

    return predicted_tags
