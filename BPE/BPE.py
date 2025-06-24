# install and import libraries
from collections import defaultdict
from transformers import AutoTokenizer


class BPE:
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.alphabet = {'</w>'}
        self.word_freqs = defaultdict(int)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = {}
        self.merges = {}
        self.initialize()

        self.vocab = self.alphabet.copy()
        
    def text_split(self, text):
        pre_tokenized_text = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        return [word for word, _ in pre_tokenized_text]

    def initialize(self):
        for text in self.corpus:
            for word in self.text_split(text):
                self.word_freqs[word] += 1
                self.alphabet.update(set(word))

        self.split = {word: list(word) + ['</w>'] for word in self.word_freqs.keys()}

    def print_split(self):
        max_len = max(len(word) for word in self.split)
        for word, s in self.split.items():
            padding = " " * (max_len - len(word))
            print(f"{word}{padding} → {' '.join(s)}")

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            word_split = self.split[word]
            for i in range(len(word_split) - 1):
                pairs[(word_split[i], word_split[i + 1])] += freq
        return pairs

    def merge_pair(self, pair):
        self.merges[pair] = ''.join(pair)
        for word in self.word_freqs:
            split = self.split[word]
            if len(split) == 1:
                continue
            idx = 0
            while idx < len(split) - 1:
                if (split[idx], split[idx + 1]) == pair:
                    split[idx] = ''.join(pair)
                    del split[idx + 1]
                else:
                    idx += 1

    def find_single_item(self, item):
        for split in self.split.values():
            for i in split:
                if i == item:
                    return True
        return False

    def train(self):
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats()
            if not pairs:
                break
            sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            max_pair, max_freq = sorted_pairs[0]
            if max_freq <= 1:
                break

            self.merge_pair(max_pair)
            self.vocab.add(''.join(max_pair))
            for i in max_pair:
                if not self.find_single_item(i):
                    self.vocab.remove(i)
            print("Vocab size: ", len(self.vocab), end='\r')

    def tokenize(self, text):
        pre_tokenized_text = self.text_split(text)
        splits_text = [[_ for _ in word] for word in pre_tokenized_text]

        for pair in self.merges.keys():
            for split in splits_text:
                idx = 0
                while idx < len(split) - 1:
                    if (split[idx], split[idx + 1]) == pair:
                        split[idx] = ''.join(pair)
                        del split[idx + 1]
                    else:
                        idx += 1
        result = sum(splits_text, [])
        return result

    def export_vocab(self, vocab_path="vocab.json"):
        vocab = {token: idx for idx, token in enumerate(self.vocab)}

        import json
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab to {vocab_path}")

    def export_merges(self, merges_path="merges.txt"):
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair in self.merges.keys():
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"Saved merges to {merges_path}")