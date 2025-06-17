import math

samples = [ [2.1, 0.5, -1.3], [0.2, 0.2, 0.2], [5.0, 0.1, 0.0] ]

def softmax(logits):
    probs = []
    logits_max = max(logits)
    logit_exps = [math.exp(logit-logits_max) for logit in logits]
    probs = [logit_exp/sum(logit_exps) for logit_exp in logit_exps]
    return probs

def entropy(probs):
    result = 0
    for prob in probs:
        item_entropy = prob * (math.log2(prob))
        result -= item_entropy
    return result

entropy_samples = []
for sample in samples:
    entropy_samples.append((sample, entropy(softmax(sample))))
print(sorted(entropy_samples, key=lambda x: -x[1]))
