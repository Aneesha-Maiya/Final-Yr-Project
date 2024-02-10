from datasets import load_dataset

dataset = load_dataset("EdinburghNLP/xsum")
# print(dataset)
print(dataset['train'])
print(type(dataset))
print(len(dataset))
print(dataset.shape)
print(dataset.num_columns)
print(dataset.num_rows)
print(dataset.column_names)
print(dataset['train'][0].items())