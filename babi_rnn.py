import re
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import layers, models


directory = '../Datasets/tasks_1-20_v1-2/en-10k'
MAX_STORY_LENGTH = 100
MAX_QUES_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
RECURRENT_LAYER = layers.recurrent.LSTM
STORY_HIDDEN = 32
QUEST_HIDDEN = 32
EPOCHS = 5
TASK = 10


def get_file_list(data_type: str):
    filename = []
    pattern = re.compile(r"{}\.txt".format(data_type))
    for _, _, files in os.walk(directory):
        filename = [file for file in files if re.search(pattern, file)]
        filename = sorted(filename, key=lambda f: int(re.match(r'qa(\d+).*', f).group(1)))
    return filename


def tokenize(sentence):
    return [word.strip() for word in re.split(r'(\W+)', sentence.lower()) if word.strip()]


def structure_data(filename: str):
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
    answers = tokenizer.texts_to_sequences(answer)
    supporting_facts = tokenizer.texts_to_sequences(facts)
    if only_supporting:
        stories = supporting_facts
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


if __name__ == '__main__':
    train_files = get_file_list('train')
    test_files = get_file_list('test')
    train_context, train_question, train_answer, train_sup_facts = structure_data(train_files[TASK - 1])
    test_context, test_question, test_answer, test_sup_facts = structure_data(test_files[TASK - 1])
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(train_context + train_question)
    # print(tokenizer.word_index)
    # extracting max length of story and question from train and test data
    # instead of keeping it default since test story/question length might be longer
    MAX_STORY_LENGTH = max(map(len, train_context + test_context))
    MAX_QUES_LENGTH = max(map(len, train_question + test_question))
    x_train, q_train, y_train = convert_data(train_context, train_question, train_answer, train_sup_facts,
                                             only_supporting=True)
    # print(x_train[0], q_train[0], y_train[0])
    x_test, q_test, y_test = convert_data(test_context, test_question, test_answer, test_sup_facts,
                                          only_supporting=True)
    # print(x_train.shape, q_train.shape, y_train.shape)
    # print(x_test.shape, q_test.shape, y_test.shape)

    word_embeddings = get_word_embeddings(tokenizer.word_index)

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
    combined_model.summary()
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, q_train.shape, y_train.shape, len(tokenizer.word_index) + 1, MAX_STORY_LENGTH, MAX_QUES_LENGTH)
    combined_model.fit([x_train, q_train], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)

    loss, acc = combined_model.evaluate([x_test, q_test], y_test)

    print("Test loss: {}, test accuracy: {}".format(loss, acc))
