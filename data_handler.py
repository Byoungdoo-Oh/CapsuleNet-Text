# coding: utf-8
from konlpy.tag import Komoran
import numpy as np

tagger = Komoran()

def load_nsmc(dataset):
	examples = open(dataset, 'r', encoding='utf-8')

	data = list()
	labels = list()
	max_len = 0

	for sentence in examples:
		try:
			d = tagger.morphs(sentence.strip().split('\t')[1])
			l = int(sentence.strip().split('\t')[-1].strip())
			length = len(d)

			if length > max_len:
				max_len = length

			data.append(d)
			labels.append(l)
		except:
			print(sentence)

	# 	d = tagger.morphs(sentence.strip().split('\t')[1])
	# 	l = int(sentence.strip().split('\t')[-1].strip())
		# data.append(d)
		# labels.append(l)

	return np.array(data), np.array(labels), max_len
	# data = [tagger.morphs(s.strip().split('\t')[1].strip()) for s in train_example]
	# print(train_data[0])
	# label = [int(s.strip().split('\t')[-1].strip()) for s in train_example]
	# # for s in train_data:
	# # 	print(tagger.morphs(s))
	# # 	break
	# # train2morphs = [tagger.morphs(s) for s in train_data]
	# # print(train2morphs[0])
	# print(train_label[0])

	# test_example = list(open(test, 'r', encoding='utf-8'))
	# test_data = [s.strip().split('\t')[1].strip() for s in test_example]
	# test_label = [int(s.strip().split('\t')[-1].strip()) for s in test_example]



# train = './nsmc/ratings_train.txt'
# test = './nsmc/ratings_test.txt'
# # load_nsmc(train)
# data, labels, vocab_size, max_len = load_nsmc(train)
# print(data[0])
# print(labels[0])
# print(vocab_size)
# print(max_len)