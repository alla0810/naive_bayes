# test_submission.py
import csv
from submission import Solution

data_path = "zoo.data.txt"  

X_train, Y_train = [], []
with open(data_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # animal_name, 16 features (0~15), class_type = 17
        features = list(map(float, row[1:-1]))  # hair~catsize
        label = int(row[-1])
        X_train.append(features)
        Y_train.append(label)

X_test = X_train[-5:]
Y_test_true = Y_train[-5:]  
X_train = X_train[:-5]
Y_train = Y_train[:-5]


model = Solution()

priors = model.prior(X_train, Y_train)
preds = model.label(X_train, Y_train, X_test)


print("=== Prior Probabilities ===")
print(priors)
print("\n=== Predictions on Test Set ===")
for i, (pred, true) in enumerate(zip(preds, Y_test_true)):
    print(f"Sample {i+1}: predicted={pred}, true={true}")
