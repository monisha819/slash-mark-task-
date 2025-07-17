from textblob import TextBlob

# Function to check polarity
def check_polarity(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Polarity ranges from -1 (neg) to 1 (pos)

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment

# Example
text_input = input("Enter a sentence to check its polarization: ")
polarity, sentiment = check_polarity(text_input)

print(f"\nPolarity Score: {polarity}")
print(f"Sentiment: {sentiment}")
