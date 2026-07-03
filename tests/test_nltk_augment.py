import nlpaug.augmenter.word as naw
import nltk
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
except:
    pass

augmenter = naw.SynonymAug(aug_src="wordnet")
print(f"Original: 'Sentence A.'")
print(f"Augmented: {augmenter.augment(['Sentence A.'])}")
