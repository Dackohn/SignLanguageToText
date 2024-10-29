from collections import deque, Counter
import time
from spellchecker import SpellChecker  # If using pyspellchecker

# Initialize the spell checker
spell = SpellChecker()

# Buffer and smoothing settings
buffer_size = 5
prediction_buffer = deque(maxlen=buffer_size)

# Word and character buffering settings
word_buffer = []
recognized_words = []
pause_threshold = 1.0
last_prediction_time = time.time()


def smooth_prediction(new_prediction):
    prediction_buffer.append(new_prediction)
    most_common = Counter(prediction_buffer).most_common(1)
    return most_common[0][0]


def buffer_letter(letter):
    global last_prediction_time

    current_time = time.time()

    if current_time - last_prediction_time > pause_threshold:
        if word_buffer:
            recognized_words.append("".join(word_buffer))
            word_buffer.clear()

    word_buffer.append(letter)
    last_prediction_time = current_time


def correct_words(words):
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None:  # Handle NoneType
            corrected_words.append(word)
        else:
            corrected_words.append(corrected_word)
    return corrected_words


def construct_sentence(words):
    sentence = " ".join(words)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    if sentence and not sentence.endswith(('.', '!', '?')):
        sentence += '.'

    return sentence


def post_process_predictions(predictions):
    smoothed_predictions = [smooth_prediction(pred) for pred in predictions]
    for letter in smoothed_predictions:
        buffer_letter(letter)

    if word_buffer:
        recognized_words.append("".join(word_buffer))

    corrected_words = correct_words(recognized_words)
    sentence = construct_sentence(corrected_words)

    return sentence


# Example of using the pipeline with a sequence of predictions
example_predictions = ['h', 'h', 'e', 'e', 'l', 'l', 'l', 'o', 'o', '_', '_', 'w', 'w', 'o', 'o', 'r', 'r', 'l', 'l', 'd']
final_sentence = post_process_predictions(example_predictions)
print("Polished Text:", final_sentence)
