# Submit this file to Gradescope
import math
from typing import Dict, List, Tuple
# You may use any built-in standard Python libraries
# You may NOT use any non-standard Python libraries such as numpy, scikit-learn, etc.

num_C = 7 # Represents the total number of classes

class Solution:
  
  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """
        Calculate prior probabilities P(y=c) with Laplace smoothing (alpha = 0.1).
        P(y=c) = (count_c + 0.1) / (N + 0.1 * num_C)
    """
    N = len(Y_train)
    alpha = 0.1

    counts = [0] * num_C
    for y in Y_train:
      counts[y - 1] += 1   # labels are 1..7

    priors = []
    for c in range(num_C):
      priors.append((counts[c] + alpha) / (N + alpha * num_C))

    return priors
    

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
        """
        Predict labels for X_test using Multinomial Naive Bayes with Laplace smoothing (alpha = 0.1)
        P(x_i = f | y=c) = (count + 0.1) / (class_count + 0.1 * num_unique_values)
        """
        alpha = 0.1

        # ----- Step 1: Count class frequencies -----
        class_counts = [0] * num_C
        for y in Y_train:
            class_counts[y - 1] += 1

        # ----- Step 2: Prepare attribute domains -----
        # Each index: number of unique values for that attribute
        # Order is exactly as dataset:
        # hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
        # backbone, breathes, venomous, fins, legs, tail, domestic, catsize
        attr_unique = [
            2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2,
            6,  # legs: possible values = {0,2,4,5,6,8}
            2, 2, 2
        ]

        # ----- Step 3: Count feature occurrences per class -----
        # feature_counts[c][i][value] = count of (feature i = value) for class c
        feature_counts = []
        for _ in range(num_C):
            feature_counts.append([{} for _ in range(len(attr_unique))])

        for x, y in zip(X_train, Y_train):
            c = y - 1
            for i, value in enumerate(x):
                if value not in feature_counts[c][i]:
                    feature_counts[c][i][value] = 0
                feature_counts[c][i][value] += 1

        # ----- Step 4: Prior probabilities -----
        N = len(Y_train)
        priors = [(class_counts[c] + alpha) / (N + alpha * num_C) for c in range(num_C)]

        # ----- Step 5: Predict labels for test data -----
        results = []

        for x in X_test:
            log_probs = []

            for c in range(num_C):
                if class_counts[c] == 0:
                    log_prob = -1e18  # avoid impossible class
                else:
                    log_prob = math.log(priors[c])

                for i, value in enumerate(x):
                    count_f = feature_counts[c][i].get(value, 0)
                    denom = class_counts[c] + alpha * attr_unique[i]
                    likelihood = (count_f + alpha) / denom
                    log_prob += math.log(likelihood)

                log_probs.append(log_prob)

            best_class = min([i for i, v in enumerate(log_probs) if v == max(log_probs)]) + 1

            results.append(best_class)

        return results
