# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Calculate metrics for the LSTM model
accuracy_1 = accuracy(y_pred_lstm, y_test)
precision_1 = precision(y_pred_lstm, y_test)
recall_1 = recall(y_pred_lstm, y_test)
f1_1 = f1(y_pred_lstm, y_test)
print("LSTM Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_1, precision_1, recall_1, f1_1))

# Calculate metrics for the GRU model
accuracy_2 = accuracy(y_pred_gru, y_test)
precision_2 = precision(y_pred_gru, y_test)
recall_2 = recall(y_pred_gru, y_test)
f1_2 = f1(y_pred_gru, y_test)
print("GRU Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_2, precision_2, recall_2, f1_2))
