from babi_rnn import get_file_list, structure_data, TASK

TASK = 1
directory = '../Datasets/tasks_1-20_v1-2/en'
train_files = get_file_list('train', directory)
contexts, questions, answers, facts = structure_data(train_files[TASK - 1], True)
print(contexts[:5])
print(questions[:5])
print(answers[:5])
print(facts[:5])