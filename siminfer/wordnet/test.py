from nltk.corpus import wordnet as wn

word = 'earthquake'
for synset in wn.synsets(word):
    words = synset.lemma_names()
    for w in words:
        if w == word or '_' in w:
            continue
        scores = []
        for s in wn.synsets(w):
            score = synset.path_similarity(s)
            if score is not None:
                scores.append(score)
        if not scores:
            continue
        print(w, sum(scores)/len(scores))
