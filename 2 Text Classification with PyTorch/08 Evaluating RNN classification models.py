# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Generate the predictions
outputs = rnn_model(X_test_seq)
_, predicted = torch.max(outputs, 1)

# Calculate the metrics
accuracy_score = accuracy(predicted, y_test_seq)
precision_score = precision(predicted, y_test_seq)
recall_score = recall(predicted, y_test_seq)
f1_score = f1(predicted, y_test_seq)
print("RNN Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_score, precision_score, recall_score, f1_score))
