import nlpaug.augmenter.word as naw
from datasets import Dataset

texts = ["The quick brown fox jumps over the lazy dog.", "Another test sentence."]
augmenter = naw.SynonymAug(aug_src="wordnet")

print(f"Input type: {type(texts)}")
results = augmenter.augment(texts)
print(f"Results type: {type(results)}")
print(f"Results: {results}")

ds = Dataset.from_dict({"text": texts})
texts_from_ds = list(ds["text"])
print(f"Input from DS type: {type(texts_from_ds)}")
results_ds = augmenter.augment(texts_from_ds)
print(f"Results from DS: {results_ds}")
