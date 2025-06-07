from textblob import TextBlob

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment

    print(f"\nText: {text}")
    print(f"Polarity: {sentiment.polarity}")
    print(f"Subjectivity: {sentiment.subjectivity}")

    if sentiment.polarity > 0:
        print("Sentiment: Positive 😊")
    elif sentiment.polarity < 0:
        print("Sentiment: Negative 😠")
    else:
        print("Sentiment: Neutral 😐")

# Main program
if __name__ == "__main__":
    user_input = input("Enter text to analyze sentiment: ")
    analyze_sentiment(user_input)
