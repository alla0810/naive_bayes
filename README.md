# Decision Tree Project (CS412)

This project implements a simple **binary decision tree classifier**  
in pure Python as part of the **CS412 Data Mining** coursework.

The tree:

- uses **entropy** and **information gain** as the splitting criterion,
- handles **continuous-valued attributes** via mid-point split candidates,
- has **maximum depth = 2** (root depth = 0, so up to 3 levels of nodes).


## Project Overview

- **Goal**: Train a shallow decision tree on numeric features and  
  classify test instances using the learned tree structure.

- **Core ideas**:
  - Entropy \(Info(D)\)
  - Split information \(Info_A(D)\)
  - Information Gain \(Gain(A) = Info(D) - Info_A(D)\)
  - Recursive tree construction with a depth limit


## Implementation Details

All functions are implemented in `submission.py` inside a single class:

```python
class Solution:
    def split_info(self, data, label, split_dim, split_point): ...
    def fit(self, train_data, train_label): ...
    def classify(self, train_data, train_label, test_data): ...
```

### 1. `split_info(data, label, split_dim, split_point)`

- Computes the **expected information** after splitting dataset \(D\)  
  into two subsets based on:

  - feature index: `split_dim`
  - threshold: `split_point`

- Split rule:

  - Left child:  instances with `x[split_dim] <= split_point`
  - Right child: instances with `x[split_dim] > split_point`

- For each side, compute entropy:

  
$$    
  Info(D_i) = - \sum_{c} p_{i,c} \log_2 p_{i,c}
$$  
  

- Then:

  $$
  Info_A(D) = \frac{|D_L|}{|D|} Info(D_L) +
              \frac{|D_R|}{|D|} Info(D_R)
  $$

- This value is used in `fit()` to compute information gain:

  $$
  Gain(A) = Info(D) - Info_A(D)
  $$


### 2. `fit(train_data, train_label)`

- Builds the decision tree and stores the root in `self.root`.

- Steps:

  1. Create a root `Node`.
  2. Recursively search for the **best split** (feature + threshold)
     using **information gain** over:
     - all feature dimensions, and
     - candidate split points defined as midpoints between sorted
       distinct feature values.
  3. Stopping conditions (leaf node):
     - all labels in the node are identical, or
     - current depth reaches the **maximum depth (2)**, or
     - no split yields positive information gain, or
     - one side of the split would be empty.

- At every node, `node.label` is set to the **majority class** among
  the labels at that node. In case of a tie, the **smaller label value**
  is chosen.

- Tie-breaking for splits:
  - If multiple splits have the same gain, choose the one with the
    **smaller feature index**.
  - If the same feature has multiple candidate thresholds with
    identical gain, pick the **smaller split_point**.


### 3. `classify(train_data, train_label, test_data)`

- First calls `fit(train_data, train_label)` to build the tree.
- Then, for each test instance:

  1. Start from `self.root`.
  2. While the node is not a leaf (`node.split_dim != -1`):
     - If `x[split_dim] <= split_point` → go to `node.left`
     - Else → go to `node.right`
  3. Return `node.label` at the leaf.

- Returns a list of integer predictions for all test instances.


## Node Structure

The `Node` class (provided by the assignment) is used to store tree structure:

```python
class Node:
    def __init__(self):
        self.split_dim = -1
        self.split_point = -1
        self.label = -1
        self.left = None
        self.right = None
```

- **Internal nodes**:
  - `split_dim >= 0`
  - `split_point` is a float threshold
  - `left`, `right` are child `Node` objects

- **Leaf nodes**:
  - `split_dim = -1`
  - `split_point = -1.0`
  - `left = right = None`


## Testing

Several helper scripts are used to test the implementation:

- `test_split_info_batch.py`
  - Uses files under `./split_info/`
  - Verifies `split_info()` against expected values.

- `text_classification_batch.py`
  - Uses files under `./classification/`
  - Verifies `classify()` predictions against expected labels.

- `test_tree_structure_batch.py`
  - Uses files under `./tree_structure/`
  - Compares preorder & inorder traversals of the built tree to
    expected outputs (tree structure check).

Example:

```bash
python test_split_info_batch.py
python text_classification_batch.py
python test_tree_structure_batch.py
```


## Notes

- Implementation uses **pure Python** (no NumPy, no scikit-learn).
- The code is designed to be compatible with the course autograder,
  which imports `Solution` from `submission.py` and calls the methods
  directly.

© 2025 by KyoSook Shin
