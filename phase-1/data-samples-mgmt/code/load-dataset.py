from datasets import load_dataset

# dataset = load_dataset('imdb')
# print(dataset)
# print(dataset['train'][0])

# dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

# for i, example in enumerate(dataset):
#     print(example["title"])
#     if i >= 4:
#         break

from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")


dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")