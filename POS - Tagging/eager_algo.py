from math import log, exp

# Eager algorithm
def eager_algorithm(sent, transition_probs, emission_probs):
    start_tag = '<s>'
    predicted_tags = [start_tag]

    # Initialize the previous tag as the start tag
    prev_tag = start_tag

    for token in sent:
        word = token['form']

        # Calculate the probability of each tag given the previous tag and the word
        max_prob = float('-inf')
        max_tag = None
        for tag in transition_probs[prev_tag].samples():
            # Calculate the probability using both transition and emission probabilities
            trans_prob = log(transition_probs.get(prev_tag).prob(tag))

            # If the word is not in the emission probabilities, skip it
            if (not emission_probs.get(tag)):
                continue
            ems_prob = log(emission_probs.get(tag).prob(word))
            # ems_prob = log(emission_probs[tag].prob(word))
            prob = trans_prob + ems_prob
            prob = exp(prob)

            # Update the maximum probability and the tag with the maximum probability
            if prob > max_prob:
                max_prob = prob
                max_tag = tag

        # Update the previous tag and add the predicted tag to the list
        prev_tag = max_tag
        predicted_tags.append(max_tag)

    # Add the end tag
    predicted_tags.append('</s>')

    return predicted_tags