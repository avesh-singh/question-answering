import re
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import layers, models

import tensorflow as tf
# print(tf.config.experimental.list_physical_devices('GPU'))

directory = '../Datasets/tasks_1-20_v1-2/en'
MAX_STORY_LENGTH = 100
MAX_QUES_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
RECURRENT_LAYER = layers.recurrent.LSTM
STORY_HIDDEN = 32
QUEST_HIDDEN = 32
EPOCHS = 5
TASK = 6


def get_file_list(data_type: str, direc=directory):
    filename = []
    pattern = re.compile(r"{}\.txt".format(data_type))
    for _, _, files in os.walk(direc):
        filename = [file for file in files if re.search(pattern, file)]
        filename = sorted(filename, key=lambda f: int(re.match(r'qa(\d+).*', f).group(1)))
    return filename


def tokenize(sentence):
    return [word.strip() for word in re.split(r'(\W+)', sentence.lower()) if word.strip()]


def structure_data(filename: str, complete=True):
    contexts = []
    questions = []
    answers = []
    supporting_facts = []
    print(filename)
    with open(directory + '/' + filename, 'r') as f:
        data = f.readlines()
        pattern = re.compile(r'(\d+) ([A-z .,]+)')
        ques_pattern = re.compile(r'(\d+) ([A-z ?,]+)\t([A-z,]+)\t([0-9 ]+)')
        story = []
        for line in data:
            is_ques = False
            if '?' not in line:
                found = re.findall(pattern, line)[0]
            else:
                is_ques = True
                found = re.findall(ques_pattern, line)[0]
            if int(found[0]) == 1:
                story = tokenize(found[1])
                # print(story)
                scenario = []
            elif is_ques:
                contexts.append(story.copy())
                questions.append(tokenize(found[1]))
                answers.append(tokenize(found[2]))
                support_lines = map(int, found[3].split())
                supporting_facts.extend([scenario[l - 1] for l in support_lines])
                if not complete:
                    story = []
            else:
                story.extend(tokenize(found[1]))
            scenario.append(tokenize(found[1]))
    print(contexts[:5])
    print(questions[:5])
    print(answers[:5])
    return contexts, questions, answers, supporting_facts


def one_hot_encoding(seq, vocab_size):
    results = np.zeros((len(seq), vocab_size + 1))
    for i, s in enumerate(seq):
        results[i, s] = 1
    return results


def convert_data(story, question, answer, facts, only_supporting=False):
    stories = tokenizer.texts_to_sequences(story)
    questions = tokenizer.texts_to_sequences(question)
    answers = np.asarray(tokenizer.texts_to_sequences(answer))
    if only_supporting:
        stories = tokenizer.texts_to_sequences(facts)
    return (
        pad_sequences(stories, MAX_STORY_LENGTH),
        pad_sequences(questions, MAX_QUES_LENGTH, padding='post'),
        one_hot_encoding(answers, len(tokenizer.word_index))
    )


def get_word_embeddings(word_index):
    with open("../../glove.6B/glove.6B.100d.txt", 'r') as g:
        embeddings = g.readlines()
        required_words = word_index.keys()
        embedding_matrix = np.zeros((len(required_words) + 1, EMBEDDING_DIM))
        for e in embeddings:
            line_split = e.split(' ')
            word = line_split[0]
            embedding = line_split[1:]
            vector = np.asarray(embedding, dtype='float32')
            if word in required_words:
                embedding_matrix[word_index[word]] = vector
        return embedding_matrix


def reconstruct_sentence(seq, mapping):
    reverse_mapping = {value: key for key, value in mapping.items()}
    return ' '.join([reverse_mapping.get(s, '?') for s in seq])


def train_model():
    # story representation model
    story = layers.Input(shape=(MAX_STORY_LENGTH,), dtype='int32')
    story_model = layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=EMBEDDING_DIM,
        weights=[word_embeddings],
        input_length=MAX_STORY_LENGTH,
        trainable=False)(story)
    story_model = RECURRENT_LAYER(STORY_HIDDEN, activation='sigmoid')(story_model)
    # story_model = layers.Dense(32, activation='relu')(story_model)

    question = layers.Input(shape=(MAX_QUES_LENGTH,), dtype='int32')
    question_model = layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=EMBEDDING_DIM,
        weights=[word_embeddings],
        input_length=MAX_QUES_LENGTH,
        trainable=False)(question)
    question_model = RECURRENT_LAYER(QUEST_HIDDEN, activation='sigmoid')(question_model)
    # question_model = layers.Dense(32, activation='relu')(question_model)

    combined_repr = layers.concatenate([story_model, question_model])
    probs = layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')(combined_repr)
    combined_model = models.Model([story, question], probs)
    # combined_model.summary()
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, q_train.shape, y_train.shape, len(tokenizer.word_index) + 1, MAX_STORY_LENGTH, MAX_QUES_LENGTH)
    combined_model.fit([x_train, q_train], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)

    loss, acc = combined_model.evaluate([x_test, q_test], y_test)
    print("Test loss: {}, test accuracy: {}".format(loss, acc))

    return combined_model


if __name__ == '__main__':
    train_files = get_file_list('train')
    test_files = get_file_list('test')
    train_context, train_question, train_answer, train_sup_facts = structure_data(train_files[TASK - 1], False)
    test_context, test_question, test_answer, test_sup_facts = structure_data(test_files[TASK - 1], False)
    if TASK == 19:
        # for task 19, the answers only contain letters not words, thus mapping them
        task19_mapping = {'s': 'south', 'e': 'east', 'w': 'west', 'n': 'north'}
        for i in range(len(train_answer)):
            train_answer[i] = list(map(lambda x: task19_mapping.get(x, ','), train_answer[i]))
        for i in range(len(test_answer)):
            test_answer[i] = list(map(lambda x: task19_mapping.get(x, ','), test_answer[i]))
        print(train_answer[0])
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(train_context + train_question + train_answer)
    print(tokenizer.word_index)
    # extracting max length of story and question from train and test data
    # instead of keeping it default since test story/question length might be longer
    MAX_STORY_LENGTH = max(map(len, train_context + test_context))
    MAX_QUES_LENGTH = max(map(len, train_question + test_question))
    x_train, q_train, y_train = convert_data(train_context, train_question, train_answer, train_sup_facts,
                                             only_supporting=False)
    # print(x_train[0], q_train[0], y_train[0])
    x_test, q_test, y_test = convert_data(test_context, test_question, test_answer, test_sup_facts,
                                          only_supporting=False)
    # print(x_train.shape, q_train.shape, y_train.shape)
    # print(x_test.shape, q_test.shape, y_test.shape)

    # load glove vectors for to words required and put them in a matrix
    word_embeddings = get_word_embeddings(tokenizer.word_index)

    model = train_model()
    prediction = model.predict([x_train, q_train])
    print("Testing")
    # print([if for i in range(y_train[0]) > 0)
    print("Story: {}\nQuestion: {}\nAnswer: {}\nModel Answer: {}".format(
        reconstruct_sentence(x_train[0], tokenizer.word_index),
        reconstruct_sentence(q_train[0], tokenizer.word_index),
        np.argsort(-y_train[0])[:5],
        np.argsort(-prediction[0])[:5]))
