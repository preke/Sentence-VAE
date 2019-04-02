import json


with open('dumps/2019-Apr-01-16:33:42/valid_E50.json', 'r') as file:
    sentences = json.load(file)

target = sentences['target_sents'][:20]
gen = sentences['gen_sents'][:20]
for i in range(len(target)):
	print(target[i])
	print(gen[i])
	print('\n')