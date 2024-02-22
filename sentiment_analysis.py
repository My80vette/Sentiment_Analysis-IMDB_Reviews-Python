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

# Load the IMDB dataset with metadata and labels
data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# Now we will pull the data and split it up into 2 groups, training and testing data
# The data object returned by tfds.load() has 2 keys, train and test
data_train, data_test = data['train'], data['test']
"""
    This was used to see an example output to format the preprocessing function
    example_batch = data_train.take(1) 
    for example in tfds.as_numpy(example_batch):
        print(example) 
    print('above this')
    exit(1)
"""


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

# dataset.map is our workhorse, it applies the "preprocess_text" function to the entire dataset
# "num_parallel_calls=tf.data.AUTOTUNE" is going to optimize the process by enabling parallelization
def preprocess_dataset(dataset, batch_size=32):
    dataset = dataset.batch(batch_size)
    def extract_text_and_label(example, _):
        raw_text = tf.numpy_function(tf.io.decode_raw, [example[0], tf.uint8], tf.string)  
        text = tf.reshape(raw_text, ()) 
        encoded_text = tokenizer(text.numpy().decode('utf-8'))  
        return encoded_text['input_ids'], encoded_text['attention_mask'], example[1] 
    return dataset.map(extract_text_and_label, num_parallel_calls=tf.data.AUTOTUNE)


# Lets preprocess the training and test data and return a tensorflow dataset
preprocessed_data_train = preprocess_dataset(data_train)
preprocessed_data_test = preprocess_dataset(data_test)

print(preprocessed_data_test.take(1))

example_batch = data_train.take(1)
for example in example_batch.as_dict():  # Iterate to get a single dictionary
    print(example) 
    print(type(example)) 