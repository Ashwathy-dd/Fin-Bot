import sys

# Load BERT model and tokenizer
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Process input
user_input = ' '.join(sys.argv[1:])
inputs = tokenizer(user_input, return_tensors='pt')

# Generate response
outputs = model(**inputs)
predicted_label = outputs.logits.argmax().item()

# Print response
print("Bot:", predicted_label)
