import os
import tempfile

from data import JsonLinesCorpus


test_data_tokens = [
    [[0,5], [1,20], [2,1], [3,30]],
    [[4,1], [3,25], [5,2]],
    [[6,1], [7,3], [2,2]],
    [[8,1]],
]
test_data_dicts = [{'id': i, 'tokens': tokens} for i, tokens in enumerate(test_data_tokens)]


def test_corpus_iter():
    with tempfile.TemporaryDirectory() as tmpdirname:
        corpus = build_corpus(tmpdirname, data=test_data_tokens)
        assert len(corpus) == 4
        assert list(corpus)[2] == test_data_tokens[2]
        assert all(tokens == test_data_tokens[i] for i, tokens in enumerate(corpus))


def test_corpus_iter_dict():
    with tempfile.TemporaryDirectory() as tmpdirname:
        corpus = build_corpus(tmpdirname)
        assert len(corpus) == 4
        assert list(corpus.iter_all())[2] == test_data_dicts[2]
        assert all(tokens == test_data_dicts[i] for i, tokens in enumerate(corpus.iter_all()))


def test_corpus_subset():
    with tempfile.TemporaryDirectory() as tmpdirname:
        corpus = build_corpus(tmpdirname)
        assert len(corpus) == 4
        file_sub = os.path.join(tmpdirname, 'corpus-sub.json')
        corpus_sub = corpus.subset(file_sub, lambda x: x['id'] > 1)
        assert len(corpus_sub) == 2
        assert corpus_sub[1] == test_data_tokens[3]
        assert all(tokens == test_data_dicts[i+2] for i, tokens in enumerate(corpus_sub.iter_all()))


def test_corpus_compressed():
    with tempfile.TemporaryDirectory() as tmpdirname:
        corpus = build_corpus(tmpdirname, fname='corpus.json.bz2')
        assert len(corpus) == 4
        assert all(tokens == test_data_dicts[i] for i, tokens in enumerate(corpus.iter_all()))
        file_sub = os.path.join(tmpdirname, 'surpus-sub.json.gz')
        corpus_sub = corpus.subset(file_sub, lambda x: x['id'] > 1)
        assert len(corpus_sub) == 2
        assert corpus_sub[1] == test_data_tokens[3]


def build_corpus(tmp_dir, fname='corpus.json', data=test_data_dicts) -> JsonLinesCorpus:
    fname = os.path.join(tmp_dir, fname)
    JsonLinesCorpus.serialize(fname, data)
    return JsonLinesCorpus(fname)
