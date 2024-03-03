import spacy
from textblob import TextBlob
import pandas as pd

nlp = spacy.load('en_core_web_sm')

# Load dataset
reviews_df = pd.read_csv("/Users/elzbietafurgalska/cogrammar/capstone/amazon_product_reviews.csv")

# Remove empty observations
clean_data =  reviews_df.dropna(subset=['reviews.text'])

# Transfer dataset to a list
reviews_data = clean_data['reviews.text'].tolist()

# Check it's worked
print(reviews_data)

# Create a function for calculating polarity
def analyze_polarity(review):
    """
    Analyzes the polarity of a given review using spaCy and TextBlob.

    Parameters:
    - review (str): The text of the review.

    Returns:
    float: The polarity score of the review.
    """
    # Preprocess the text with spaCy
    doc = nlp(review)
    # Analyze sentiment with TextBlob
    blob = TextBlob(review)
     
    return blob.sentiment.polarity

# Create a function to preprocess data by removing stop words and changing to lower case
def preprocess_reviews(review):
    """
    Preprocesses a review by converting it to lowercase and removing stop words.

    Parameters:
    - review (str): The text of the review.

    Returns:
    str: The preprocessed review text without stop words.
    """
    doc = nlp(review.lower())
    cleaned_review_words = [token.text for token in doc if not token.is_stop]

    return ' '.join(cleaned_review_words)

# Test with some sample reviews
sample_reviews = [reviews_data[10], reviews_data[20], reviews_data[30]]

sample_number = 0
for review in sample_reviews:
    cleaned_sample = preprocess_reviews(review)
    polarity = analyze_polarity(cleaned_sample)
    
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    sample_number += 1
    print(f"Sample {sample_number}: {review}\nPolarity: {polarity}\nSentiment: {sentiment}\n\n")


# Now, iterate the functioning model through the full list of reviews, using both functions
review_number = 0
for review in reviews_data:
    cleaned_review = preprocess_reviews(review)
    polarity = analyze_polarity(cleaned_review)
    
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    review_number += 1
    print(f"Review {review_number}: {review}\nPolarity: {polarity}\nSentiment: {sentiment}\n\n")


# Calculating similarity, using a bigger spaCy model as the reviews are quite long
nlp = spacy.load('en_core_web_md')

# Randomly chosen reviews, compared for similarity
my_review_of_choice_1 = nlp(reviews_df['reviews.text'][3])
my_review_of_choice_2 = nlp(reviews_df['reviews.text'][45])
similarity = my_review_of_choice_1.similarity(my_review_of_choice_2)

print(f'''Random review 1: {my_review_of_choice_1}\nRandom review 2: {my_review_of_choice_2}\nSimilarity: {similarity}''')
