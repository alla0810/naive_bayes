import math
from typing import List

num_C = 7

class Solution:

    def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
        alpha = 0.1
        N = len(Y_train)
        class_counts = [0] * num_C
        for y in Y_train:
            class_counts[y - 1] += 1
        total_classes = len([c for c in class_counts if c > 0])
        priors = [(class_counts[c] + alpha) / (N + alpha * total_classes) for c in range(num_C)]
        return priors

    def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
        alpha = 0.1

        # Count per class
        class_counts = [0] * num_C
        for y in Y_train:
            class_counts[y - 1] += 1

        # Feature domain sizes
        attr_unique = [2]*12 + [9] + [2]*3

        # Conditional counts
        feature_counts = [[{} for _ in range(len(attr_unique))] for _ in range(num_C)]
        for x, y in zip(X_train, Y_train):
            c = y - 1
            for i, val in enumerate(x):
                v = int(val)
                feature_counts[c][i][v] = feature_counts[c][i].get(v, 0) + 1

        # Priors
        N = len(Y_train)
        total_classes = len([c for c in class_counts if c > 0])
        priors = [(class_counts[c] + alpha) / (N + alpha * total_classes) for c in range(num_C)]

        # Predict
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
            preds.append(max(best_classes) + 1)
        return preds
