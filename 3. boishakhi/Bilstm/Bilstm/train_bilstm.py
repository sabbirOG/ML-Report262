import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


# Step 1: Load datasets
train_df = pd.read_csv("output/train.csv")
test_df = pd.read_csv("output/test.csv")

print(train_df.head())
print(test_df.head())

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


# Step 2: Separate input and labels

X_train = train_df["text"].astype(str)
y_train = train_df["label"]

X_test = test_df["text"].astype(str)
y_test = test_df["label"]

print("\nFirst 5 training SMS:")
print(X_train.head())

print("\nFirst 5 training labels:")
print(y_train.head())

print("\nFirst 5 testing SMS:")
print(X_test.head())

print("\nFirst 5 testing labels:")
print(y_test.head())


# Step 3: Tokenization

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

# Fit ONLY on training data
tokenizer.fit_on_texts(X_train)

# Convert text to numerical sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Make every sequence the same length for the BiLSTM model
MAX_LENGTH = 100
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)
X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# Check tokenization
print("\nOriginal SMS:")
print(X_train.iloc[0])

print("\nTokenized SMS:")
print(X_train_seq[0])

print("\nPadded SMS shape:")
print(X_train_pad.shape)
print(X_test_pad.shape)


# Step 9: Create the BiLSTM model
model = Sequential()


# Step 10: Convert token IDs into dense word vectors
model.add(
    Embedding(
        input_dim=10000,
        output_dim=128
    )
)


# Step 11: Process the sequence in both directions
model.add(
    Bidirectional(
        LSTM(64)
    )
)


# Step 12: Reduce overfitting during training
model.add(
    Dropout(0.5)
)


# Step 13: Output one probability for binary classification
model.add(
    Dense(1, activation="sigmoid")
)


# Step 15: Display the model architecture
model.build((None, MAX_LENGTH))
model.summary()


# Step 16: Configure the model for binary classification
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Step 17: Train the model
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


# Step 18: Evaluate the trained model on unseen test data
test_loss, test_accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Step 19: Convert predicted probabilities into binary labels
y_probability = model.predict(X_test_pad)
y_pred = (y_probability >= 0.5).astype(int)

print("\nFirst 5 predicted probabilities:")
print(y_probability[:5].flatten())

print("\nFirst 5 predicted labels:")
print(y_pred[:5].flatten())


# Step 20: Calculate prediction accuracy
accuracy = accuracy_score(
    y_test,
    y_pred.flatten()
)

print("\nAccuracy:", accuracy)


# Step 21: Calculate precision for spam predictions
precision = precision_score(
    y_test,
    y_pred.flatten()
)

print("Precision:", precision)


# Step 22: Calculate recall for spam predictions
recall = recall_score(
    y_test,
    y_pred.flatten()
)

print("Recall:", recall)


# Step 23: Calculate the F1 score
f1 = f1_score(
    y_test,
    y_pred.flatten()
)

print("F1 Score:", f1)


# Step 24: Display all metrics together
print("\nAll Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# Step 25: Create the confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred.flatten()
)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# Step 26: Plot the confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()


# Step 27: Plot training and validation accuracy
plt.figure()
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()


# Step 28: Plot training and validation loss
plt.figure()
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# Step 29: Save the trained model and tokenizer
model.save("output/bilstm_sms_spam_model.keras")

with open("output/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("\nModel saved to output/bilstm_sms_spam_model.keras")
print("Tokenizer saved to output/tokenizer.pkl")


# Step 30: Test the model with a new SMS
new_sms = [
    "Congratulations! You have won a free prize."
]

new_seq = tokenizer.texts_to_sequences(new_sms)
new_pad = pad_sequences(
    new_seq,
    maxlen=100,
    padding="post",
    truncating="post"
)

prediction = model.predict(new_pad)

if prediction[0][0] >= 0.5:
    print("New SMS prediction: Spam")
else:
    print("New SMS prediction: Ham")
