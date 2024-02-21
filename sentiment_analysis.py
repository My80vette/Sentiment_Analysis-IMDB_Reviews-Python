"""
    Nicholas Capriotti
    2/24/24
    IMDB_Review Sentiment Analysis
    This is a simple project to perform a sentiment analysis on various IMDB reviews, the goal is to
    understand the foundation of NLP and apply it to a relevant project, our goal is to achieve 90% accuracy
    We will use Tensorflow documentation as well as Google Gemini to act as our interactive tutor
"""
# Importing tensorflow to act as the backbone for our models capabilities, and the necessary dataset (IMDB_Reviews)
import tensorflow as tf
import tensorflow_datasets as tfds
# This will use Huggingface to pull the relevant tokenizer for our preprocessing based on the one we select (DistilBERT)
from transformers import AutoTokenizer

# tfds.load pulls the relevant dataset with metadata about the dataset, and it pulls the sentament labels so we can train and test
data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# Now we will pull the data and split it up into 2 groups, training and testing data
# The data object returned by tfds.load() has 2 keys, train and test
data_train, data_test = data['train'], data['test']

"""
    We will now move on to preprocessing
    This process will allow us to tokenize the plaintext from the dataset, meaning break it up into individual words or units
    We will then do some 'cleaning', meaning we will remove punctuation, make everything lowercase, and remove words with little 
    meaning such as 'the' & 'and'.
    We will use DistilBERT-base-uncased to tokenize our text without caring about uppercase or lowercase letters
    This model should be successful for our simplistic use case
"""
# The AutoTokenizer will download the relevant tokenizer model from Huggingface based on the name given
tokenizer = AutoTokenizer.from_pretrained("elastic/distilbert-base-uncased-finetuned-conll03-english")

"""
    We will now define a function to take the review and have the DistilBERT tokenizer perform tokenization and cleaning
    The input function takes a review and its associated label
    we add the "review.decode('utf-8')" to ensure compatability with various text encodings
    It returns the tokenized result for our model
"""
def preprocess_text(review, label):
    encoded_review = tokenizer(review.decode('utf-8'))
    return encoded_review
