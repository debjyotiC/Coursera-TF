import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = ['I love my dog',
             'I love my cat',
             'You love my dog!']
             # 'Do you think my dog is amazing?']

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
# sequences = tokenizer.texts_to_sequences(sentences)
#
# test_data = ['I really love my dog',
#              'My dog loves my manatee']
#
# test_seq = tokenizer.texts_to_sequences(test_data)
# padded = pad_sequences(sequences, padding='post')
# print(word_index)
# print(sequences)
# print(padded)
