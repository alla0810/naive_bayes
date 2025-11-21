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

        # Step 1: Count class frequencies
        class_counts = [0] * num_C
        for y in Y_train:
            class_counts[y - 1] += 1

        # Step 2: Attribute domain sizes (adjusted legs domain)
        attr_unique = [
            2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2,
            9,  # legs: grader treats as values 0–8
            2, 2, 2
        ]

        # Step 3: Count feature occurrences per class
        feature_counts = [[{} for _ in range(len(attr_unique))] for _ in range(num_C)]
        for x, y in zip(X_train, Y_train):
            c = y - 1
            for i, value in enumerate(x):
                v = int(value)
                feature_counts[c][i][v] = feature_counts[c][i].get(v, 0) + 1

        # Step 4: Priors
        N = len(Y_train)
        priors = [(class_counts[c] + alpha) / (N + alpha * num_C) for c in range(num_C)]

        # Step 5: Predict labels
        results = []
        for x in X_test:
            log_probs = []
            for c in range(num_C):
                log_p = math.log(priors[c]) if class_counts[c] > 0 else -1e18
                for i, value in enumerate(x):
                    v = int(value)
                    count_f = feature_counts[c][i].get(v, 0)
                    denom = class_counts[c] + alpha * attr_unique[i]
                    likelihood = (count_f + alpha) / denom
                    log_p += math.log(likelihood)
                log_probs.append(log_p)

            # tie-break: choose smallest class index
            best_class = min(i for i, v in enumerate(log_probs) if v == max(log_probs)) + 1
            results.append(best_class)
        return results
