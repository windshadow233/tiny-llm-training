from BPE import BPE

# import the Wikipedia corpus used for training
with open('wiki_corpus.txt', encoding="utf8") as f:
    corpus = f.readlines()
    print(corpus[:5])

# set the hyperparameter of vocabulary size
vocab_size = 3000

# create a BPE tokenizer object
MyBPE = BPE(corpus=corpus, vocab_size=vocab_size)

# train BPE tokenizer with Wikipedia corpus
MyBPE.train()

MyBPE.export_vocab()
MyBPE.export_merges()