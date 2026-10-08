from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline(
    "text-classification",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model"
)

# New messages for prediction
messages = [
    "My package has not arrived yet",
    "I want to get my money back",
    "I want to send this product back",
    "I want to change my order",
]

# Predict each message
for message in messages:

    result = classifier(message)

    print("\nMessage:", message)
    print("Prediction:", result[0]["label"])
    print("Confidence:", round(result[0]["score"], 4))