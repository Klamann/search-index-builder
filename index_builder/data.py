"""
a collection of data structures used throughout this project
"""
import json
import logging
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, Any, Iterable, Callable
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy
from gensim.corpora import IndexedCorpus
from gensim.utils import SlicedCorpus

import util

logger = logging.getLogger('index-builder')

# type declarations
CorpusDoc = Union[Dict[str,Any], Iterable[Tuple[int,float]]]
TopicTokens = List[Tuple[str, float]]


class JsonLinesCorpus(IndexedCorpus):
    """
    All corpus formats in gensim are vastly over-engineered, yet at the same time they
    fail to fulfil the most basic purposes:
    - how do we identify documents, if we can't store document IDs?
    - what about additional metadata we'd like to store?
    - why use parsers written with the python-skills of a 5-year-old?

    The sane solution to all these issues is to use python dicts to store data,
    and have a proper convention how to name stuff. Python's json parser has
    proper C bindings that will outperform the custom crap all the other parsers
    use anyways...

    Here's the convention:
    - one document per line
    - each line consists of a single json dictionary (no line breaks within the json)
    - the dictionary has at least these fields:
      * "id": the identifier of the document. defaults to null
      * "tokens": a list of (token_id, count) tuples
    - arbitrary data can be stored under any other key

    Other features
    - can read and write compressed corpora (restriction: no random access, only iteration is available)
    """

    def __init__(self, fname: str):
        """
        Initialize the corpus from an existing file.
        """
        IndexedCorpus.__init__(self, fname)
        logger.info("loading corpus from %s" % fname)
        self.fname = fname
        self.length = None

    def __iter__(self) -> Iterable[List[Tuple[int, float]]]:
        """
        Iterate over the corpus, returning one sparse vector at a time.
        This iterator is basically there fore compatibility with gensim models.
        To iterate over the documents with metadata, use `iter_all()` instead
        :return: a generator that yields lists of (token id, count) tuples
        """
        i = 0
        with util.open_by_ext(self.fname, 'rt', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                yield self.line2doc(line)
        self.length = i + 1

    def iter_all(self) -> Iterable[Dict[str, Any]]:
        """
        Iterate over the corpus, returning a document with all additional metadata
        as dictionary. Tokens are stored under 'tokens'. The document ID is
        expected under 'id', but not required.
        Alternatively, use `__iter__` to get sparse token vectors.
        :return: a generator that yields each document as dictionary.
        """
        with util.open_by_ext(self.fname, 'rt', encoding='utf-8') as fp:
            for line in fp:
                yield json.loads(line)

    @staticmethod
    def save_corpus(fname: str, corpus: Iterable[CorpusDoc], id2word: Dict[int,str] = None,
                    labels: List[str] = None, progress_cnt: int = 10000, metadata=False):
        """
        Save a corpus in the JsonLinesCorpus format.
        This function is automatically called by `JsonLinesCorpus.serialize`;
        don't call it directly, call `serialize` instead.
        :param fname: the file name of the corpus to store
        :param corpus: the actual corpus. An iterable of documents. A document can be either a
               list of (token, count) tuples, or a dictionary that must have at least a 'tokens' field
        :param id2word: a mapping from token id to token string.
                if specified, all string tokens are converted to integers using this mapping
                and all tokens that are not found in this mapping are dropped!
        :param labels: a mapping from document index to document label. can be used to store
               document IDs, although the canonical way would be to store the document ID under
               the key 'id' in the dictionary.
        :param progress_cnt: log progress information every n documents
        :param metadata: an unused parameter that was reserved by the base class...
        :return: the offsets of each document
        """
        logger.info("converting corpus to JsonLinesCorpus format: %s" % fname)
        token2id = {v: k for k, v in id2word.items()} if id2word else None
        offsets = []
        with util.open_by_ext(fname, 'wt', encoding='utf-8') as fp:
            use_offsets = util.is_stream_seekable(fp)
            for i, doc in enumerate(corpus):
                if use_offsets:
                    offsets.append(fp.tell())
                label = labels[i] if labels else None
                fp.write(JsonLinesCorpus.doc2line(doc, label=label, token2id=token2id))
                if progress_cnt and ((i+1) % progress_cnt) == 0:
                    logger.info("%i documents have been processed so far" % (i+1))
        return offsets

    @classmethod
    def serialize(cls, fname: str, corpus: Iterable[CorpusDoc], id2word: Dict[int,str] = None,
                  index_fname: str = None, progress_cnt: int = 10000, labels: List[str] = None,
                  metadata=False):
        """
        Iterate through the document stream `corpus`, saving the documents to `fname`
        and recording byte offset of each document. Save the resulting index
        structure to file `index_fname` (or `fname`.index is not set).

        The index can be stored in compressed form, if the '.bz2' extension (for bzip2)
        or '.gz' extension (for gzip) are used. In this case, no index will be written
        and random access will be slower.

        >>> JsonLinesCorpus.serialize('test.json', corpus)
        >>> corpus = JsonLinesCorpus('test.json') # document stream now has random access
        >>> print(corpus[42]) # retrieve document no. 42, etc.
        """
        if getattr(corpus, 'fname', None) == fname:
            raise ValueError("identical input vs. output corpus filename, refusing to serialize: %s" % fname)
        if fname.endswith('.bz2') or fname.endswith('.gz'):
            # bypass default serialization routine, just save the corpus
            cls.save_corpus(fname, corpus, id2word=id2word, progress_cnt=progress_cnt, labels=labels)
        else:
            # use the default serialization routine, which also generates a corpus index
            super().serialize(fname, corpus, id2word=id2word, index_fname=index_fname,
                              progress_cnt=progress_cnt, labels=labels, metadata=metadata)

    def convert_tokens_to_ids(self, fname: str, id2word: Dict[int,str], progress_cnt: int = 10000):
        """
        convert the plaintext tokens in this corpus to token IDs, based on the specified dictionary.
        warning: all tokens that are not part of this dictionary will be lost!
        do not apply this function, if this is already a reduced corpus.
        use has_plain_tokens() to check if it contains plaintext tokens.
        """
        self.serialize(fname, self.iter_all(), id2word=id2word, progress_cnt=progress_cnt)

    def has_plain_tokens(self) -> bool:
        """
        returns True, iff this corpus has plaintext tokens (strings) instead of token ids.
        this decision is made based on a small sample (first document)
        """
        sample_tokens = self.__iter__().__next__()
        return all(isinstance(token, str) for token,count in sample_tokens)

    def subset(self, fname: str, filter_function: Callable[[Dict[str, Any]], bool],
               progress_cnt: int = 10000) -> 'JsonLinesCorpus':
        """
        Generate a subset of this corpus. Provide a filter function that
        decides, which documents to keep. The filter has access to the entire
        document dictionary.
        Example: `lambda doc: doc['id'].startswith("foo")`
        :param fname: the name of the file where the new corpus will be written to.
        :param filter_function: this function should return True for every document to keep.
        :param progress_cnt: log progress information every n documents
        :return: a subset of this corpus whose documents match the specified filter condition
        """
        filtered_corpus = (doc for doc in self.iter_all() if filter_function(doc))
        JsonLinesCorpus.serialize(fname, filtered_corpus, progress_cnt=progress_cnt)
        return JsonLinesCorpus(fname)

    def docbyoffset(self, offset):
        """
        Return the document stored at file position `offset`.
        """
        with util.open_by_ext(self.fname, 'rt', encoding='utf-8') as fp:
            fp.seek(offset)
            return self.line2doc(fp.readline())

    def docbyindex(self, index):
        with util.open_by_ext(self.fname, 'rt', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i == index:
                    return self.line2doc(line)

    def _count_lines(self):
        """
        counts the lines of the file, without parsing it's contents
        """
        with util.open_by_ext(self.fname, 'rt', encoding='utf-8') as fp:
            return sum(1 for _ in fp)

    @staticmethod
    def line2doc(line: str) -> List[Tuple[int, float]]:
        """
        Create a document from a single line (string)
        :param line: a line of the corpus file
        :return: the document as a list of (token id, count) tuples
        """
        tokens = json.loads(line).get('tokens')
        return list(tokens.items()) if isinstance(tokens, dict) else tokens

    @staticmethod
    def doc2line(doc: CorpusDoc, label: Any = None, token2id: Dict[str,int] = None):
        """
        Output the document in SVMlight format, as a string. Inverse function to `line2doc`.
        """
        if isinstance(doc, dict):
            if 'tokens' not in doc:
                raise ValueError("the specified dictionary has no 'tokens' field: %s" % str(doc))
            result = doc
        else:
            # assume (token, count) tuples, take ID from label parameter
            result = {'id': label, 'tokens': doc}
        if token2id:
            tokens = list(result['tokens'].items()) if isinstance(result['tokens'], dict) else result['tokens']
            result['tokens'] = [(token2id[token], count) for token, count in tokens if token in token2id]
        return json.dumps(result, separators=None, indent=None) + '\n'

    def __len__(self):
        """
        Return the index length if the corpus is indexed. Otherwise, make a pass
        over self to calculate the corpus length and cache this number.
        """
        if self.index is not None:
            return len(self.index)
        if self.length is None:
            self.length = self._count_lines()
            return self.length
        return self.length

    def __getitem__(self, docno):
        if isinstance(docno, (slice, list, numpy.ndarray)):
            return SlicedCorpus(self, docno)
        elif isinstance(docno, (int, numpy.integer)):
            if self.index is not None:
                return self.docbyoffset(self.index[docno])
            else:
                return self.docbyindex(docno)
        else:
            raise ValueError('Unrecognised value for docno, use either a single integer, a slice or a numpy.ndarray')

    def __str__(self) -> str:
        return "JsonLinesCorpus(fname={}, length={})".format(self.fname, len(self))

    def __repr__(self) -> str:
        return self.__str__()


class Topic:

    def __init__(self, topic_id: str, tokens: TopicTokens = None, layer: int = 1,
                 parent: 'Topic' = None, children: List['Topic'] = None, **kwargs):
        """
        initializes a new topic
        :param topic_id: the ID of this topic
        :param tokens: a list of (token, score) tuples, in descending score order
        :param layer: this topic's layer (depth in the tree, 0 for root node, 1 for
               root's children, and so on)
        """
        super().__init__()
        self.topic_id = topic_id
        self.tokens = tokens
        self.layer = layer
        self.parent = parent
        self.children = children

    def add_child(self, topic_id: str, tokens: TopicTokens = None):
        """
        adds a new child to this topic. automatically infers the topic layer
        and adds references between parent and child
        :param topic_id: the ID of the child topic
        :param tokens: a list of (token, score) tuples, in descending score order
        :return: the new child topic
        """
        layer = 1 if self.layer is None else self.layer+1
        child = Topic(topic_id, tokens, layer=layer, parent=self)
        if self.children is None:
            self.children = []
        self.children.append(child)
        return child

    def store_recursively(self) -> List[Dict[str, Any]]:
        """
        store this topic and all of it's descendants in an easily serializable
        dictionary structure without object references
        """
        topics = self._collect_topics()
        return self.store_topics(topics)

    def _collect_topics(self) -> List['Topic']:
        """
        :return: a list containing this topic and all of it's descendants
        """
        topics = [self]
        if self.children:
            for child in self.children:
                topics.extend(child._collect_topics())
        return topics

    @staticmethod
    def store_topics(topics: Iterable['Topic']) -> List[Dict[str, Any]]:
        """
        transform topics into an easily serializable dictionary structure without object references
        :param topics: a list of topics
        :return: a list of the dictionary representation of each topic (see `to_dict`)
        """
        return [topic.to_dict() for topic in topics]

    @staticmethod
    def restore_topics(topic_dicts: List[Dict[str, Any]]) -> Dict[str, 'Topic']:
        """
        restore a bunch of topics that have been transformed by `store_topics`.
        Also restores object references between topic objects.
        :param topic_dicts: a list of topic dictionaries
        :return: a dictionary mapping from topic id to topic object
        """
        topic_map = {td['topic_id']: Topic._from_dict(td) for td in topic_dicts}
        for topic in topic_map.values():
            topic._restore_links(topic_map)
        return topic_map

    def to_dict(self):
        """
        transforms this topic into an easily serializable dictionary structure without object references
        :return: this topic's fields as dictionary, without object references
        """
        return OrderedDict([
            ('topic_id', self.topic_id),
            ('layer', self.layer),
            ('tokens', self.tokens),
            ('parent', self.parent.topic_id if self.parent else None),
            ('children', [child.topic_id for child in self.children] if self.children else None),
        ])

    @staticmethod
    def _from_dict(topic_dict: Dict[str, Any]) -> 'Topic':
        """
        restores a Topic object from it's dictionary representation.
        Does NOT restore links to other Topic objects. To restore a
        serialized topic model, please refer to `restore_topics`
        :param topic_dict:
        :return:
        """
        return Topic(**topic_dict)

    def _restore_links(self, topic_map: Dict[str, 'Topic']):
        if self.parent:
            self.parent = topic_map[self.parent]
        if self.children:
            self.children = [topic_map[child_id] for child_id in self.children]

    def __hash__(self):
        return hash(self.topic_id) * hash(self.layer)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.topic_id == other.topic_id and self.layer == other.layer
        return False

    def __str__(self) -> str:
        return util.class_str_from_dict("Topic", self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()


class Document:

    def __init__(self, doc_id: str, topics: Dict[Topic, float] = None, categories: List[str] = None, **kwargs):
        """
        initializes a new document
        :param doc_id: the unique identifier of this document
        :param topics: a mapping from topic id to score
        :param categories: the (optional) category of this document
        """
        super().__init__()
        self.doc_id = doc_id
        self.topics = topics
        self.categories = categories

    def add_topic(self, topic: Topic, score: float):
        if self.topics is None:
            self.topics = {}
        self.topics[topic] = score

    def top_topics(self, min_layer: int = None) -> List[Tuple[Topic, float]]:
        """
        get the highest scoring topics for this document
        :param min_layer: only include topics of this layer or higher
        :return: a list of (topic, score) tuples, in descending order of score
        """
        topics = [(topic, score) for topic, score in self.topics.items() if
                  (min_layer is None or topic.layer >= min_layer)]
        return sorted(topics, key=lambda x: x[1], reverse=True)

    @staticmethod
    def store_documents(documents: Iterable['Document']) -> List[Dict[str, Any]]:
        """
        transform documents into an easily serializable dictionary structure without object references
        :param documents: a list of topics
        :return: a list of the dictionary representation of each document (see `to_dict`)
        """
        return [doc.to_dict() for doc in documents]


    @staticmethod
    def restore_documents(doc_dicts: List[Dict[str, Any]], known_topics: Dict[str, 'Topic'] = None) -> Dict[str, 'Document']:
        """
        restore a bunch of documents that have been transformed by `store_documents`.
        Also restores object references from documents to topics.
        :param doc_dicts: a list of document dictionaries
        :return: a dictionary mapping from document id to document object
        """
        return {doc_dict['doc_id']: Document.from_dict(doc_dict, known_topics) for doc_dict in doc_dicts}

    def to_dict(self):
        return OrderedDict([
            ('doc_id', self.doc_id),
            ('categories', self.categories),
            ('topics', [(topic.topic_id, score) for topic, score in self.topics.items()] if self.topics else None),
        ])

    @staticmethod
    def from_dict(doc_dict: Dict[str, Any], known_topics: Dict[str, Topic] = None) -> 'Document':
        topics = None
        if known_topics and doc_dict['topics']:
            topics = {known_topics[topic_id]: score for topic_id, score in doc_dict['topics']}
        return Document(doc_dict['doc_id'], topics=topics, categories=doc_dict['categories'])

    def __str__(self) -> str:
        return util.class_str_from_dict("Document", self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()


class DocumentCollection:
    """
    a document collection consists of
    - a topic definition, with the most relevant tokens for each topic
    - a list of documents, each containing a reference to one or more topics
    """

    def __init__(self, topics: Dict[str, Topic] = None, documents: Dict[str, Document] = None):
        """
        initializes a new document collection
        :param topics: mapping from topic id to Topic (object)
        :param documents: mapping from document id to Document (object)
        """
        super().__init__()
        self.topics = topics
        self.documents = documents

    def to_dict(self):
        return OrderedDict([
            ('topics', Topic.store_topics(self.topics.values())),
            ('documents', [doc.to_dict() for id, doc in self.documents.items()]),
        ])

    @staticmethod
    def from_dict(doc_collection: Dict[str, Any]) -> 'DocumentCollection':
        raw_topics = doc_collection['topics']
        raw_documents = doc_collection['documents']
        topics = Topic.restore_topics(raw_topics)
        documents = {doc_dict['doc_id']: Document.from_dict(doc_dict, topics) for doc_dict in raw_documents}
        return DocumentCollection(topics, documents)

    def __str__(self) -> str:
        return "DocumentCollection(topics: {}, documents: {})".format(
            len(self.topics) if self.topics else 0, len(self.documents) if self.documents else 0)

    def __repr__(self) -> str:
        return self.__str__()
