import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare the training data
training_data = [
    ("This is a sentence.", "positive"),
    ("This is another sentence.", "negative"),
]


# Tokenize the training data
tokenized_training_data = []

for sentence, label in training_data:
    encoded_inputs = tokenizer(sentence, return_tensors="pt")
    tokenized_training_data.append((encoded_inputs, label))

# Create a dataloader for the training data
dataloader = torch.utils.data.DataLoader(
    tokenized_training_data, batch_size=8, shuffle=True
)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the model
model.save_pretrained("./trained_bert_base.bin")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")

# Evaluate the model
test_data = [
    ("This is a test sentence.", "positive"),
    ("This is another test sentence.", "negative"),
]

tokenized_test_data = []
for sentence, label in test_data:
    encoded_inputs = tokenizer(sentence, return_tensors="pt")
    tokenized_test_data.append((encoded_inputs, label))

test_dataloader = torch.utils.data.DataLoader(
    tokenized_test_data, batch_size=8, shuffle=False
)

correct = 0
total = 0

for batch in test_dataloader:
    inputs, labels = batch
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()
    total += len(labels)

accuracy = correct / total
print("Accuracy:", accuracy)
