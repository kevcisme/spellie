# encoding: utf-8
  
from collections import Counter
import re
import numpy as np
from numpy.random import choice as random_choice
from numpy.random import randint as random_randint
from numpy.random import shuffle as random_shuffle
from numpy.random import rand
from numpy import zeros as np_zeros  # pylint:disable=no-name-in-module
from time import time

# Parameters for the model and dataset
MAX_INPUT_LEN = 40
MIN_INPUT_LEN = 3
AMOUNT_OF_NOISE = 0.2 / MAX_INPUT_LEN
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)  # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(
                                    re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)),
                                re.UNICODE)


class DataSet(object):
    """
    Loads news articles from a file, generates misspellings and vectorizes examples.
    """

    def __init__(self, dataset_filename, test_set_fraction=0.1, inverted=True):
        self.inverted = inverted

        news = self.read_news(dataset_filename)
        questions, answers = self.generate_examples(news)

        chars_answer = set.union(*(set(answer) for answer in answers))
        chars_question = set.union(*(set(question) for question in questions))
        self.chars = sorted(list(set.union(chars_answer, chars_question)))
        self.character_table = CharacterTable(self.chars)

        split_at = int(len(questions) * (1 - test_set_fraction))
        (self.questions_train, self.questions_dev) = (questions[:split_at], questions[split_at:])
        (self.answers_train, self.answers_dev) = (answers[:split_at], answers[split_at:])

        self.x_max_length = max(len(question) for question in questions)
        self.y_max_length = max(len(answer) for answer in answers)

        self.train_set_size = len(self.questions_train)
        self.dev_set_size = len(self.questions_dev)

        print("Completed pre-processing")

    def train_set_batch_generator(self, batch_size):
        return self.batch_generator(self.questions_train, self.answers_train, batch_size)

    def dev_set_batch_generator(self, batch_size):
        return self.batch_generator(self.questions_dev, self.answers_dev, batch_size)

    def batch_generator(self, questions, answers, batch_size):
        start_index = 0

        while True:
            questions_batch = []
            answers_batch = []

            while len(questions_batch) < batch_size:
                take = min(len(questions) - start_index, batch_size - len(questions_batch))

                questions_batch.extend(questions[start_index: start_index + take])
                answers_batch.extend(answers[start_index: start_index + take])

                start_index = (start_index + take) % len(questions)

            yield self.vectorize(questions_batch, answers_batch)

    def add_noise_to_string(self, a_string, amount_of_noise):
        """Add some artificial spelling mistakes to the string"""
        if rand() < amount_of_noise * len(a_string):
            # Replace a character with a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
        if rand() < amount_of_noise * len(a_string):
            # Delete a character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
        if len(a_string) < MAX_INPUT_LEN and rand() < amount_of_noise * len(a_string):
            # Add a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
        if rand() < amount_of_noise * len(a_string):
            # Transpose 2 characters
            random_char_position = random_randint(len(a_string) - 1)
            a_string = (a_string[:random_char_position] +
                        a_string[random_char_position + 1] +
                        a_string[random_char_position] +
                        a_string[random_char_position + 2:])
        return a_string

    def vectorize(self, questions, answers):
        """Vectorize the questions and expected answers"""

        assert len(questions) == len(answers)

        X = np_zeros((len(questions), self.x_max_length, self.character_table.size), dtype=np.bool)

        for i in range(len(questions)):
            sentence = questions[i]
            for j, c in enumerate(sentence):
                X[i, j, self.character_table.char_indices[c]] = 1

        y = np_zeros((len(answers), self.y_max_length, self.character_table.size), dtype=np.bool)

        for i in range(len(answers)):
            sentence = answers[i]
            for j, c in enumerate(sentence):
                y[i, j, self.character_table.char_indices[c]] = 1

        return X, y

    def clean_text(self, text):
        """Clean the text - remove unwanted chars, fold punctuation etc."""

        text = text.strip()
        text = NORMALIZE_WHITESPACE_REGEX.sub(' ', text)
        text = RE_DASH_FILTER.sub('-', text)
        text = RE_APOSTROPHE_FILTER.sub("'", text)
        text = RE_LEFT_PARENTH_FILTER.sub("(", text)
        text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
        text = RE_BASIC_CLEANER.sub('', text)

        return text

    def read_news(self, dataset_filename):
        """Read the news corpus"""
        print("Reading news")
        news = open(dataset_filename, encoding='utf-8').read()
        print("Read news")

        lines = [line for line in news.split('\n')]
        print("Read {} lines of input corpus".format(len(lines)))

        lines = [self.clean_text(line) for line in lines]
        print("Cleaned text")

        counter = Counter()
        for line in lines:
            counter += Counter(line)

        most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
        print(most_popular_chars)

        lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
        print("Left with {} lines of input corpus".format(len(lines)))

        return lines

    def generate_examples(self, corpus):
        """Generate examples of misspellings"""

        print("Generating examples")

        questions, answers, seen_answers = [], [], set()

        while corpus:
            line = corpus.pop()

            while len(line) > MIN_INPUT_LEN:
                if len(line) <= MAX_INPUT_LEN:
                    answer = line
                    line = ""
                else:
                    space_location = line.rfind(" ", MIN_INPUT_LEN, MAX_INPUT_LEN - 1)
                    if space_location > -1:
                        answer = line[:space_location]
                        line = line[len(answer) + 1:]
                    else:
                        space_location = line.rfind(" ")  # no limits this time
                        if space_location == -1:
                            break  # we are done with this line
                        else:
                            line = line[space_location + 1:]
                            continue

                if answer and answer in seen_answers:
                    continue

                seen_answers.add(answer)
                answers.append(answer)

        print('Shuffle')
        random_shuffle(answers)
        print("Shuffled")

        for answer_index, answer in enumerate(answers):
            question = self.add_noise_to_string(answer, AMOUNT_OF_NOISE)
            question += '.' * (MAX_INPUT_LEN - len(question))
            answer += "." * (MAX_INPUT_LEN - len(answer))
            answers[answer_index] = answer
            assert len(answer) == MAX_INPUT_LEN

            question = question[::-1] if self.inverted else question
            questions.append(question)

        print("Generated questions and answers")

        return questions, answers


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool)  
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)
