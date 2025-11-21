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
    alpha = 0.1
    N = len(Y_train)
    counts = [0] * num_C
    for y in Y_train:
      counts[y - 1] += 1
    return [(counts[c] + alpha) / (N + alpha * num_C) for c in range(num_C)]
    

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
        """
        Predict labels for X_test using Multinomial Naive Bayes with Laplace smoothing (alpha = 0.1)
        P(x_i = f | y=c) = (count + 0.1) / (class_count + 0.1 * num_unique_values)
        """
        alpha = 0.1

        # Step 1: class counts
        class_counts = [0] * num_C
        for y in Y_train:
            class_counts[y - 1] += 1

        # Step 2: attribute unique counts (legs adjusted to 9)
        attr_unique = [
            2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2,
            9,  # legs
            2, 2, 2
        ]

        # Step 3: feature occurrence counts
        feature_counts = [[{} for _ in range(len(attr_unique))] for _ in range(num_C)]
        for x, y in zip(X_train, Y_train):
            c = y - 1
            for i, val in enumerate(x):
                v = int(val)
                feature_counts[c][i][v] = feature_counts[c][i].get(v, 0) + 1

        # Step 4: priors
        N = len(Y_train)
        priors = [(class_counts[c] + alpha) / (N + alpha * num_C) for c in range(num_C)]

        # Step 5: predictions
        preds = []
        for x in X_test:
            log_probs = []
            for c in range(num_C):
                if class_counts[c] == 0:
                    log_p = -1e18
                else:
                    log_p = math.log(priors[c])

                for i, val in enumerate(x):
                    v = int(val)
                    count_f = feature_counts[c][i].get(v, 0)
                    denom = class_counts[c] + alpha * attr_unique[i]
                    likelihood = (count_f + alpha) / denom
                    log_p += math.log(likelihood)
                log_probs.append(log_p)

            max_log = max(log_probs)
            best_classes = [i for i, v in enumerate(log_probs) if abs(v - max_log) < 1e-12]
            preds.append(max(best_classes) + 1)  # tie-break to max label
        return preds
