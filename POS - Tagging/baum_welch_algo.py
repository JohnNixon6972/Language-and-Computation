from math import log, exp
from helper_functions import min_log_prob, logsumexp

def forward(sentence, transition_probs, emission_probs):
    start = '<s>'
    end = '</s>'

    alpha = [{}]

    word = sentence[0]['form']
    # Initialize when i = 1, because input don't have <s> tag
    for tag in transition_probs:

        trans_prob = min_log_prob
        if transition_probs.get(start):
            trans_prob = transition_probs.get(start).logprob(tag)

        if not emission_probs.get(tag):
            continue

        ems_prob = emission_probs.get(tag).logprob(word)

        alpha[0][tag] = trans_prob + ems_prob

    # iterate over the tokens in the sentence
    for t in range(1, len(sentence)):
        word = sentence[t]['form']

        alpha.append({})
        # iterate over the possible tags for the current token
        for curr_tag in transition_probs:
            logprob_list = []

            # iterate over the possible tags for the previous token
            for prev_tag in transition_probs:
                trans_prob = min_log_prob
                if transition_probs.get(prev_tag):
                    trans_prob = transition_probs.get(prev_tag).logprob(curr_tag)

                if not emission_probs.get(curr_tag):
                    continue

                ems_prob = emission_probs.get(curr_tag).logprob(word)

                current_prob = trans_prob + ems_prob

                if prev_tag != start:
                    logprob_list.append(
                        alpha[t - 1][prev_tag] + current_prob)
                    
            # store the maximum probability
            alpha[t][curr_tag] = logsumexp(logprob_list)

    logprob_list = []
    alpha.append({})
    # iterate over the possible tags for the end token
    for tag in transition_probs:
        trans_prob = min_log_prob
        if transition_probs.get(tag):
            trans_prob = transition_probs.get(tag).logprob(end)

        if tag not in alpha[-2]:
            continue
        logprob_list.append(alpha[-2][tag] + trans_prob)

    # store the maximum probability
    alpha[-1][end] = logsumexp(logprob_list)
    alpha.insert(0, {})
    alpha[0][start] = 0
    return alpha


def backward(sentence, transition_probs, emission_probs):
    start = '<s>'
    end = '</s>'

    beta = []
    # initialize the size of beta
    for i in range(len(sentence) + 2):
        beta.append({})

    beta[-1][end] = 0

    # iterate over the possible tags for the end token
    for prev_tag in transition_probs:
        trans_prob = min_log_prob
        if transition_probs.get(prev_tag):
            trans_prob = transition_probs.get(prev_tag).logprob(end)
        beta[-2][prev_tag] = trans_prob

    # iterate over the tokens in the sentence in reverse order
    for i in range(len(sentence) - 1, 0, -1):
        word = sentence[i + 1 - 1]['form']
        # iterate over the possible tags for the current token
        for prev_tag in transition_probs:
            logprob_list = []

            # iterate over the possible tags for the next token
            for post_tag in transition_probs:

                trans_prob = min_log_prob
                if transition_probs.get(prev_tag):
                    trans_prob = transition_probs.get(prev_tag).logprob(post_tag)

                if not emission_probs.get(post_tag):
                    continue

                ems_prob = emission_probs.get(post_tag).logprob(word)

                # calculate the probability using both transition and emission probabilities
                if post_tag != end and post_tag in beta[i + 1]:
                    logprob_list.append(
                        beta[i + 1][post_tag] + trans_prob + ems_prob)

            beta[i][prev_tag] = logsumexp(logprob_list)

    logprob_list_start = []

    # iterate over the possible tags for the start token
    word = sentence[0]['form']
    for tag_1 in transition_probs:
        trans_prob = min_log_prob
        if transition_probs.get(start):
            trans_prob = transition_probs.get(start).logprob(tag_1)

        if not emission_probs.get(tag_1):
            continue

        ems_prob = emission_probs.get(tag_1).logprob(word)

        logprob_list_start.append(
            beta[1][tag_1] + trans_prob + ems_prob)

    beta[0][start] = logsumexp(logprob_list_start)
    return beta


def individually_most_probable_tags(sent, transition_probs, emission_probs):
    # Calculate the forward and backward probabilities
    alpha = forward(sent, transition_probs, emission_probs)
    beta = backward(sent, transition_probs, emission_probs)

    predicted_tags = []
    # iterate over the tokens in the sentence
    for t in range(len(sent)):
        max_tag = None
        max_prob = min_log_prob

        # iterate over the possible tags for the current token
        for tag in transition_probs:
            if tag == '<s>' or tag == '</s>' or tag not in alpha[t + 1] or tag not in beta[t + 1]:
                continue
            prob = alpha[t + 1][tag] + beta[t + 1][tag]

            # update the maximum probability and the tag with the maximum probability
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        predicted_tags.append(max_tag)
    predicted_tags.insert(0, '<s>')
    predicted_tags.append('</s>')
    return predicted_tags