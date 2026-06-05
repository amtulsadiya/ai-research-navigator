# threshold_test.py

valid_scores = [
    0.5239,
    0.6494,
    0.6738,
    0.5853,
    0.6494,
    0.6494,
    0.5853,
    0.6494,
    0.6494,
    0.7395,
]
invalid_scores = [
    0.5853,
    0.4572,
    0.5246,
    0.5246,
    0.5246,
    0.5246,
    0.5246,
    0.5246,
    0.5246,
    0.5246,
]

best_threshold = 0
best_accuracy = 0
best_stats = {}

for t in [x / 1000 for x in range(450, 750, 5)]:
    tp = sum(1 for s in valid_scores if s >= t)
    tn = sum(1 for s in invalid_scores if s < t)
    fp = sum(1 for s in invalid_scores if s >= t)
    fn = sum(1 for s in valid_scores if s < t)

    accuracy = (tp + tn) / 20

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = t
        best_stats = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

print(f"Best threshold: {best_threshold}")
print(f"Accuracy: {best_accuracy * 100:.1f}%")
print(f"Correctly answered (TP): {best_stats['tp']}/10")
print(f"Correctly refused  (TN): {best_stats['tn']}/10")
print(f"Wrongly answered   (FP): {best_stats['fp']}/10")
print(f"Wrongly refused    (FN): {best_stats['fn']}/10")
