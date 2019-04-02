import json


with open('dumps/2019-Apr-02-02:40:54/valid_E9.json', 'r') as file:
    sentences = json.load(file)

target = sentences['target_sents'][:20]
gen = sentences['gen_sents'][:20]
for i in range(len(target)):
	print(target[i])
	print(gen[i])
	print('\n')