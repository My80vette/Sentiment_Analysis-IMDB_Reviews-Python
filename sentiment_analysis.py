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
import numpy as np
# Load the IMDB dataset with metadata and labels
data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# Now we will pull the data and split it up into 2 groups, training and testing data
# The data object returned by tfds.load() has 2 keys, train and test
data_train, data_test = data['train'], data['test']

data_train = data_train.batch(32) 
data_test = data_test.batch(32)


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
    We define a helper function decode_review. 
    It takes a TensorFlow Tensor, converts it to a NumPy array (which has decode), 
    performs the decoding, and returns a regular Python string.
"""
def preprocess_text(review, label):
    def decode_review(review_tensor):
        return review_tensor.numpy().decode('utf-8')
    # Use tf.py_function to apply decoding  
    encoded_review = tokenizer(review)
    return encoded_review


# Define a function to preprocess a single text and its label.
def preprocess_text(review, label):
    # Define a function to decode a TensorFlow Tensor to a string.
    def decode_review(review_tensor):
        return review_tensor.numpy().decode('utf-8')
    
    # Use the tokenizer to encode the review.
    encoded_review = tokenizer(review)
    return encoded_review

# Define a function to preprocess an entire dataset.
def preprocess_dataset(dataset, batch_size=32):

    # Define a function to extract text and label from each example in the dataset.
    def extract_text_and_label(example, _):
        # Extract the raw text from the example.
        raw_text = example[0]
        # Convert the raw text to a string.
        text = tf.strings.as_string(raw_text)
        # Reshape the text to ensure it's a 1D tensor.
        text = tf.reshape(text, ())

        # Define a function to tokenize the text using the tokenizer.
        def tokenize_text(text):
            return tokenizer(text.decode('utf-8'))

        # Use tf.py_function to wrap the tokenization process, allowing it to execute eagerly.
        encoded_text = tf.py_function(tokenize_text, [text], [tf.int32, tf.int32])
        # Unpack the encoded text into input IDs and attention mask.
        input_ids, attention_mask = encoded_text

        # Ensure the tensors have the correct shape.
        input_ids.set_shape([None])
        attention_mask.set_shape([None])

        # Return the input IDs, attention mask, and label as a tuple.
        return input_ids, attention_mask, example[1]

    # Apply the extract_text_and_label function to each example in the dataset.
    return dataset.map(extract_text_and_label, num_parallel_calls=tf.data.AUTOTUNE)


# Preprocess the training and test data and return a TensorFlow dataset.
preprocessed_data_train = preprocess_dataset(data_train)
preprocessed_data_test = preprocess_dataset(data_test)

# Print the first example from the preprocessed test dataset.
print(preprocessed_data_test.take(1))

# Take one example from the training dataset.
example_batch = data_train.take(1)
# Iterate over the example and print its contents and type.
for example in example_batch:
    print(example)
    print(type(example))
