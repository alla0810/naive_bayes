# Naive Bayes Classifier (CS412 — Zoo Dataset)

This project implements a **Multinomial Naive Bayes Classifier** in pure Python (no external libraries) as part of the **CS412 Data Mining** coursework.

---

## Overview

The classifier predicts the class of zoo animals based on 16 categorical and numeric attributes from the [UCI Zoo dataset](https://archive.ics.uci.edu/ml/datasets/Zoo).  
There are **7 possible classes**, representing:
1. Mammal  
2. Bird  
3. Reptile  
4. Fish  
5. Amphibian  
6. Bug  
7. Invertebrate  

---

## Files

| File | Description |
|------|--------------|
| `submission.py` | Main implementation (contains `Solution` class). |
| `test_submission.py` | Local test script for verifying implementation. |
| `zoo.data.txt` | Dataset file with 101 labeled animal samples. |
| `zoo.names.txt` | Attribute descriptions and class distribution. |
| `problem.pdf` | Official assignment instructions. |

---

## Implementation Details

All required functions are defined inside `submission.py` within the `Solution` class:

```python
class Solution:
    def prior(self, X_train, Y_train):
        ...
    def label(self, X_train, Y_train, X_test):
        ...
```

### `prior(X_train, Y_train)`

Computes **prior probabilities** \( P(y=c) \) for each class with **Laplace smoothing** (α = 0.1):

$$
P(y=c) = \frac{\text{count}(y=c) + 0.1}{N + 0.1 \times 7}
$$

Returns:  
A list `[p1, p2, ..., p7]` where ∑pi = 1.

---

### `label(X_train, Y_train, X_test)`

Implements **Multinomial Naive Bayes prediction** with Laplace smoothing (α = 0.1):

$$
\hat{y} = \arg\max_y \Big[ \log P(y) + \sum_i \log P(x_i \mid y) \Big]
$$

For each feature \( x_i \):

$$
P(x_i = f \mid y=c) = \frac{\text{count}(x_i=f, y=c) + 0.1}
{\text{count}(y=c) + 0.1 \times K_i}
$$

where \( K_i \) = number of unique values for attribute \( i \).

---

### Attribute Domains

| Attribute | Values |
|------------|---------|
| hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, tail, domestic, catsize | {0,1} |
| legs | {0,2,4,5,6,8} |

---

## Testing

Run the provided local test script:

```bash
python test_submission.py
```

Expected output example:
```
=== Prior Probabilities ===
[0.4043, 0.1975, 0.0527, 0.1355, 0.0424, 0.0734, 0.0941]

=== Predictions on Test Set ===
Sample 1: predicted=1, true=1
Sample 2: predicted=6, true=6
Sample 3: predicted=1, true=1
Sample 4: predicted=7, true=7
Sample 5: predicted=2, true=2
```

---

## Notes

- No external libraries (NumPy, pandas, scikit-learn) are used.
- Laplace smoothing (α = 0.1) is applied consistently.
- Floating-point rounding is handled automatically by the autograder.
- **Do not print** inside `submission.py` — the autograder checks only return values.

---

© 2025 by **Kyo Sook Shin**  
University of Illinois Urbana-Champaign  
CS412: Introduction to Data Mining

