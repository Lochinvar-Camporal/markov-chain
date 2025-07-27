# Advanced variable-order Markov language model with Kneser-Ney smoothing
import random
from collections import Counter, deque

import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


class AdaptiveCache:
    """Adaptive cache that boosts recently seen words."""

    def __init__(self, max_size=50, weight=0.1):
        self.max_size = max_size
        self.weight = weight
        self.cache = deque()
        self.counter = Counter()

    def update(self, token: str):
        self.cache.append(token)
        self.counter[token] += 1
        if len(self.cache) > self.max_size:
            old = self.cache.popleft()
            self.counter[old] -= 1
            if self.counter[old] <= 0:
                del self.counter[old]

    def get_distribution(self):
        total = sum(self.counter.values())
        if total == 0:
            return {}
        return {word: self.weight * count / total for word, count in self.counter.items()}


def build_language_model(sentences, order=7, prune_len=2):
    """Build a Kneser-Ney interpolated language model."""
    tokenized = [nltk.word_tokenize(s.lower()) for s in sentences]

    # Build vocabulary before fitting so smoothing statistics remain intact
    counts = Counter(word for sent in tokenized for word in sent)
    vocab = [w for w, c in counts.items() if c >= prune_len]

    train_data, _ = padded_everygram_pipeline(order, tokenized)
    model = KneserNeyInterpolated(order)
    model.fit(train_data, vocab)
    return model


def sample_next(model, context, cache=None, top_k=None, top_p=None, temperature=1.0):
    """Sample next token with top-k/nucleus and temperature."""
    context = tuple(context)
    vocab = [w for w in model.vocab if w not in {'<s>'}]
    probs = [model.score(w, context) for w in vocab]

    if cache:
        cache_dist = cache.get_distribution()
        base = 1.0 - cache.weight
        probs = [base * p + cache_dist.get(w, 0.0) for w, p in zip(vocab, probs)]

    if temperature != 1.0:
        probs = [p ** (1.0 / temperature) for p in probs]

    pairs = list(zip(vocab, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k:
        pairs = pairs[:top_k]
    if top_p:
        cumulative = 0.0
        kept = []
        for tok, p in pairs:
            cumulative += p
            kept.append((tok, p))
            if cumulative >= top_p:
                break
        pairs = kept

    tokens, values = zip(*pairs)
    total = sum(values)
    values = [p / total for p in values]
    return random.choices(tokens, weights=values, k=1)[0]


def generate_sentence(model, cache=None, order=7, top_k=None, top_p=None, temperature=1.0):
    context = ['<s>'] * (order - 1)
    result = []
    while True:
        next_word = sample_next(model, context, cache, top_k, top_p, temperature)
        if next_word == '</s>' or len(result) > 50:
            break
        result.append(next_word)
        if cache:
            cache.update(next_word)
        context = context[1:] + [next_word]
    return ' '.join(result)


def perplexity(model, sentences, order=7):
    tokenized = [nltk.word_tokenize(s.lower()) for s in sentences]
    test_data, _ = padded_everygram_pipeline(order, tokenized)
    ngrams = [ng for sent in test_data for ng in sent]
    try:
        return model.perplexity(ngrams)
    except ZeroDivisionError:
        return float('inf')


if __name__ == '__main__':
    corpus = [
        'This is an example sentence.',
        'Another sentence goes here.',
        'More data improves language model quality.'
    ]
    train = corpus[:-1]
    test = corpus[-1:]

    model = build_language_model(train, order=5)
    cache = AdaptiveCache(max_size=20, weight=0.2)

    print('Perplexity:', perplexity(model, test, order=5))
    for _ in range(3):
        print(generate_sentence(model, cache=cache, order=5, top_k=5, top_p=0.9, temperature=0.8))
