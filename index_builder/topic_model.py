import argparse
import itertools
import json
import logging
import os
import pickle
import time
import warnings
from collections import Counter, defaultdict
from typing import Dict, Any, List, Iterable, Tuple, Set

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import langdetect
import spacy
from gensim import corpora
from gensim.corpora import IndexedCorpus
from gensim.models import HdpModel, LdaMulticore
from gensim.models.basemodel import BaseTopicModel
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token

import text_processing
import util
from data import JsonLinesCorpus, Topic, Document, DocumentCollection
from util import ProgressLog

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('topic-models')
logger.setLevel(logging.INFO)


#
# goal: go from plaintext PDFs + optional metadata file (result of parser) to id-topic-mapping (input for es index)
# add optional pre-classification (id-class mapping) as first layer of the hierarchical topic model
#
# start: need corpus-id-mapping and metadata-by-doc-id.
# first layer: split by category and write multiple corpora, each with own id-mapping
# subsequent layers: split by assigned topic, add id-mapping, store topics in central id-topic-dict
# side-effect: build topic-tree, store relations in Topic objects (get parents and children, root topic is "main")
#
# so, the id-category-map is a one-time thingy that we don't need to preserve at all. Just write everything
# into the topic tree and document-topic-mapping immediately
#
# steps:
# - calculate topic model from document collection
# - classify documents using this model, store topic labels in document objects
# - create one new model per topic with different hyperparameters and train it with the sub-corpus consisting only of
#   the documents in this topic
# - recur
#
# issues:
# - need to build a new mmcorpus and a corpusindex-docid-mapping for each model
#
# data structure: LayeredTopicModel
# - recursive structure, initialize for every subsequent layer
# - the build script requires lots of state and temporary files
#   -> maybe have a separate builder, that spits out the final model...
# - the final model consists of multiple topic models + metadata in a single archive
#
# topic model visualization: https://github.com/bmabey/pyLDAvis
#


class TopicModel:

    def __init__(self, file_pdf_text: str = None, file_corpus_input: str = None,
                 file_metadata: str = None, file_output_prefix: str = None, abstracts_only=False,
                 language_filter: str = None, model: str = "hdp", batch_size=100, n_threads=None,
                 topic_layers: List[int] = None, topic_limit_per_layer: List[int] = None,
                 category_layer=False, min_docs_per_topic: int = None, token_min_count=1,
                 dict_size_limit=10000, document_limit: int = None):
        """

        :param file_pdf_text: path to the file containing the parsed PDFs (output of pdf_parser)
        :param file_corpus_input: path to the file containing the tokens of the parsed pdfs
               (optional, preferred over file_pdf_text)
        :param file_metadata: path to the metadata file (output of arxiv_crawler. required,
               if the category layer should be used)
        :param file_output_prefix: all output files, including temporary files, will be prefixed
               with this string. all results will be stored under this prefix aswell.
        :param abstracts_only: use only title and abstract for the topic model instead of the
               full document text
        :param language_filter: filter by the specified language code. the spacy parser we use
               currenlty only supports english text, so 'en' is a reasonable value here
               (though not a requirement)
        :param model: specify the model to use. supported models: "hdp", "lda"
        :param batch_size: the batch size of the spacy parser
        :param n_threads: the number of threads to use on parallelizable tasks (e.g. spacy)
        :param topic_layers: how many topics are to be calculated on each nested topic layer
        :param topic_limit_per_layer: how many of those topics should have a fixed limit during
               classification (i.e. each document can be only part of up to N topics instead of as
               many as the topic model yields)
        :param category_layer: use the categories extracted from metadata as the first layer
        :param min_docs_per_topic: how many documents are required for each sub-topic to add
               (e.g. min_docs = 100, we have 1000 documents, this limits the number of sub-topics to 10)
        :param token_min_count: lowest allowed token count for words that may appear in the dictionary
        :param dict_size_limit: the total size limit of the dictionary (take the N most frequent terms)
        :param document_limit: just process the first N documents (useful for testing)
        """
        super().__init__()
        # file paths
        self.file_pdf_text = file_pdf_text
        self.file_corpus_input = file_corpus_input
        self.file_metadata = file_metadata
        self.file_output_prefix = file_output_prefix
        # derived paths
        self.file_tasklog = file_output_prefix + '-progress.log'
        self.file_corpus_plain = file_corpus_input or file_output_prefix + '-corpus-plain.json.bz2'
        self.file_corpus = file_output_prefix + '-corpus.json'
        self.file_dict = file_output_prefix + '-lemma.dict.bz2'
        self.file_ids = file_output_prefix + '-ids.json'
        self.file_docs = file_output_prefix + '-docs.json'
        self.file_model = file_output_prefix + '-hdp.pkl.bz2'
        self.file_topics = file_output_prefix + '-topics.json.bz2'
        # application config
        self.abstracts_only = abstracts_only
        self.language_filter = language_filter
        self.model = model
        self.batch_size = batch_size
        self.n_threads = n_threads or max(2, int(os.cpu_count() / 2))
        self.topic_layers = topic_layers or [10]
        self.topic_limit_per_layer = topic_limit_per_layer or [0] * len(topic_layers)
        self.category_layer = category_layer
        self.min_docs_per_topic = min_docs_per_topic
        self.token_min_count = token_min_count
        self.dict_size_limit = dict_size_limit
        self.document_limit = document_limit
        # integrity checks
        if not abstracts_only and not file_pdf_text and not file_corpus_input:
            raise ValueError("At least one of the parameters 'file_pdf_text' or 'file_token_input' "
                             "is required, if 'abstracts_only' is not enabled.")
        if (category_layer or abstracts_only) and not file_metadata:
            raise ValueError("The parameter 'file_metadata' is required, if 'category_layer' "
                             "or 'abstracts_only' is True.")
        if not file_output_prefix:
            raise ValueError("The output path must not be empty.")

    def build(self, force=False):
        # evaluate progress information (no need to do long-running tasks twice)
        progress = ProgressLog(self.file_tasklog)
        if progress.finished:
            logger.info("skipping {} tasks that have already been finished".format(len(progress.finished)))
        # unify declarations
        if isinstance(self.topic_layers, int):
            self.topic_layers = [self.topic_layers]
        # build the corpus (if required) and vocabulary
        if force or 'token_dict' not in progress:
            self.stream_token_dict()
            progress.add('token_dict', "finished calculating the token counts and the global dictionary for all documents")
        # create a reduced version of the corpus based on the provided dictionary
        if force or 'reduced_corpus' not in progress:
            self.stream_reduced_corpus()
            progress.add('reduced_corpus', "")
        # build the category layer (if specified)
        if self.category_layer and (force or 'metadata' not in progress):
            self.stream_metadata()
            progress.add('metadata', "finished extracting categories from document metadata")
        # build the nested topic model and classify documents
        if force or 'topic_model' not in progress:
            self.stream_nested_topic_model()
            progress.add('topic_model', "")
        logger.info("build completed. Classification results have been stored in `{}`".format(self.file_topics))

    def stream_nested_topic_model(self):
        # initialize data structures
        root_topic = Topic('root', layer=0)
        current_topics = None       # type: List[Topic]
        documents = None            # type: Dict[str, Document]
        dictionary = self.load_dictionary()
        if self.category_layer:
            logger.info("building first topic layer from document metadata...")
            current_topics = self.topics_from_metadata(root_topic)
            documents = self.docs_from_metadata(current_topics)
        else:
            current_topics = [root_topic]
            documents = self.docs_from_ids()

        # build topic model and classify documents
        logger.info("building topic models and classifying documents...")
        for idx, (num_topics, topic_limit) in enumerate(zip(self.topic_layers, self.topic_limit_per_layer)):
            logger.info("Processing layer {} of {}, with {} sub-topics per parent topic{}"
                        .format(idx+1, len(self.topic_layers), num_topics, " (max. {} topics per doc)"
                                .format(topic_limit) if topic_limit else ""))

            # TODO add option to remove temporary data immediately
            # collect topics for the next iteration
            next_topics = []    # type: List[Topic]

            # go through the documents of each topic
            for topic in current_topics:
                logger.info("Processing documents in topic '{}'...".format(topic.topic_id))
                # load the last corpus that was created for this topic's parent
                corpus = self.load_corpus_for_topic(topic.parent if topic != root_topic else topic)
                # reduce the corpus so it only contains the documents we need
                sub_corpus = self.corpus2corpus(corpus, documents, topic) if topic != root_topic else corpus
                if sub_corpus:  # only continue, of there are actually documents with this topic
                    # limit the number of sub-topics, if necessary
                    num_topics_adjusted = min(int(len(sub_corpus) / self.min_docs_per_topic), num_topics) \
                        if self.min_docs_per_topic else num_topics
                    if num_topics_adjusted <= 3:
                        logger.info("skipping topic {} (too few documents: {})".format(topic.topic_id, len(sub_corpus)))
                    else:
                        # build the topic model
                        self.stream_topic_model(topic, dictionary, sub_corpus, num_topics_adjusted)
                        # classify documents using the topic model
                        sub_topics = self.stream_classify_documents(topic, sub_corpus, documents, topic_limit=topic_limit)
                        # save the sub-topics for the next layer
                        next_topics.extend(sub_topics)
                        logger.info("All {} documents in topic '{}' have been classified".format(len(sub_corpus), topic.topic_id))
                else:
                    logger.warning("there are no documents in topic '{}'. Hint: parent topic '{}' has {} documents"
                                   .format(topic.topic_id, topic.parent.topic_id if topic.parent else "root", len(corpus)))
            # select the topics for the next iteration
            current_topics = next_topics

        logger.info("all {} documents have been classified. storing results...".format(len(documents)))
        topics = {topic.topic_id: topic for topic in root_topic._collect_topics()}
        collection = DocumentCollection(topics, documents)
        util.json_write(collection.to_dict(), self.file_topics, pretty=False)

    def stream_token_dict(self):
        """
        make a single run over the file containing all documents as plaintext.
        Parse all documents using spacy, store the token counts for each document
        and build the global token dict
        """
        if self.file_corpus_input:
            logger.info("reading corpus from '{}'".format(self.file_corpus_plain))
            corpus = JsonLinesCorpus(self.file_corpus_input)
            return self.store_gensim_dict(corpus)
        else:
            if self.abstracts_only:
                logger.info("reading abstracts from '{}'".format(self.file_metadata))
                documents = util.json_read_lines(self.file_metadata, self.get_title_and_abstract)
            else:
                logger.info("reading documents from '{}'".format(self.file_pdf_text))
                documents = util.json_read_lines(self.file_pdf_text, self.combine_pages)
            # limit document count (if configured)
            documents_limited = (next(documents) for i in range(self.document_limit)) if self.document_limit else documents
            # filter by document language (if configured)
            documents_filtered = self.filter_by_lang(documents_limited, self.language_filter) if self.language_filter else documents_limited
            # parse documents using spacy
            documents_tokens = self.spacy_parse(documents_filtered, batch_size=self.batch_size, n_threads=self.n_threads)
            # stream intermediate result to disk (in case data does not fit in RAM, which it won't if you're serious about this stuff)
            return self.store_tokens_and_gensim_dict(documents_tokens)

    def stream_reduced_corpus(self):
        corpus = JsonLinesCorpus(self.file_corpus_plain)
        if corpus.has_plain_tokens():
            logger.info("building a reduced version of corpus '{}'".format(self.file_corpus_plain))
            dictionary = self.load_dictionary()
            corpus.convert_tokens_to_ids(self.file_corpus, id2word=dictionary.id2token)
        else:
            # corpus is already in reduced format. continue...
            self.file_corpus = self.file_corpus_plain

    def stream_metadata(self):
        # get the IDs of all documents we need
        documents = self.docs_from_ids()
        # read the metadata file and extract all categories for the documents we want
        logger.info("reading metadata from " + self.file_metadata)
        metadata = util.json_read_lines(self.file_metadata)     # type: List[Dict[str,Any]]
        category_count = Counter()
        for meta_dict in metadata:
            doc_id = meta_dict['header']['identifier'].split(':')[-1]
            # match doc ids
            if doc_id in documents:
                doc = documents[doc_id]
                categories = meta_dict['header']['setSpecs']
                categories_clean = sorted(set(c.split(':')[0] for c in categories))
                doc.categories = categories_clean
                for cat in categories_clean:
                    category_count[cat] += 1
        # integrity check
        for doc in documents.values():
            if doc.categories is None:
                logger.warning("there was no metadata entry for document '{}'".format(doc.doc_id))
        # reading finished. print stats and write to file
        logger.info("categories for {} documents have been read: {}".format(len(documents), category_count.items()))
        util.json_write(Document.store_documents(documents.values()), self.file_docs, pretty=False)

    def stream_topic_model(self, topic: Topic, dictionary: corpora.Dictionary = None,
                           corpus: IndexedCorpus = None, num_topics=20, max_topics_per_doc=5):
        # load dictionary and corpus, if necessary
        if not dictionary:
            dictionary = self.load_dictionary()
            logger.warning("the default dictionary was loaded from file. "
                           "You should keep an instance in memory instead of calling this in a loop...")
        if not corpus:
            corpus = JsonLinesCorpus(self.file_corpus)
            logger.warning("the default corpus was loaded from file. You should provide a "
                           "reduced corpus to increase performance (see corpus2corpus)")
        # build the model
        logger.info("building a topic model with {} topics for {} documents in topic '{}'"
                    .format(num_topics, len(corpus), topic.topic_id))
        t0 = time.time()
        if self.model == "lda":
            model = LdaMulticore(corpus, id2word=dictionary.id2token, num_topics=num_topics,
                                 passes=2, iterations=50, chunksize=2000, workers=self.n_threads)
        elif self.model == "hdp":
            # T = overall topic limit, K = max topics per document
            model = HdpModel(corpus, id2word=dictionary.id2token, T=num_topics, K=max_topics_per_doc)
        else:
            raise ValueError("Unknown model identifier '{}'".format(self.model))
        t1 = time.time()

        # serialize
        logger.info("building the model took {:.1f} s. Serializing model...".format(t1-t0))
        output_path = self._get_model_path(topic)
        with util.open_by_ext(output_path, 'wb') as fp:
            pickle.dump(model, fp, protocol=4)
            logger.info("model dump finished, took {:.1f} s".format(time.time()-t1))

    def stream_classify_documents(self, parent_topic: Topic, corpus: JsonLinesCorpus,
                                  documents: Dict[str, Document], topic_limit=0) -> List[Topic]:
        # load the actual topic model
        model = self.load_model(self._get_model_path(parent_topic))   # type: HdpModel

        # build Topic objects from model
        topics = {}
        try:
            for i in itertools.count():
                topic_id = "{}-{}".format(parent_topic.topic_id, i)
                show_topic_kwargs = {}
                if self.model == "hdp":
                    show_topic_kwargs = {'num_words': 10, 'formatted': False}
                elif self.model == "lda":
                    show_topic_kwargs = {'topn': 10}
                topic_terms = [(term, round(score, 5)) for term, score in model.show_topic(i, **show_topic_kwargs)]
                topic = parent_topic.add_child(topic_id, topic_terms)
                topics[i] = topic
        except IndexError:
            pass    # most pythonic way to interrupt iteration, if # of elements is unknown...

        # calculate the topics for each document
        logger.info("classifying {} documents from topic '{}' into {} new categories"
                    .format(len(corpus), parent_topic.topic_id, len(topics)))
        t = time.time()
        for i, doc_dict in enumerate(corpus.iter_all()):
            if not doc_dict['id'] or doc_dict['id'] not in documents:
                logger.warning("Document '{}' at corpus index {} (topic: {}) was not found "
                               "in the document index and will be skipped"
                               .format(doc_dict['id'], parent_topic.topic_id, i))
                continue
            doc_id = doc_dict['id']
            tokens = doc_dict['tokens']
            document = documents[doc_id]
            assert document.topics is None or parent_topic in document.topics, \
                "tried to classify a document which is not part of the current topic"
            doc_topics = sorted(model[tokens], key=lambda x: x[1], reverse=True)   # type: List[Tuple[str, float]]
            for topic_idx, score in (doc_topics[:topic_limit] if topic_limit else doc_topics):
                if score > 0.10:
                    document.add_topic(topics[topic_idx], round(score, 5))
            if (i+1) % 10000 == 0:
                t1 = time.time()
                logger.info("{}/{} documents have been classified ({:.2f} doc/min)"
                            .format(i+1, len(corpus), self.batch_size*60/(t1-t)))
                t = t1
        return list(topics.values())

    def corpus2corpus(self, corpus: JsonLinesCorpus, documents: Dict[str, Document], topic: Topic) -> JsonLinesCorpus:
        """
        get a subset of a corpus. It will include all documents that contain
        the specified topic.
        Writes the reduced corpus to a new file whose name is derived from the document ID
        :param corpus: the source corpus
        :param documents: the document definition (contains document topics)
        :param topic: filter all documents in the corpus by this topic
        :return: a new corpus containing only the filtered documents
        """
        logger.info("creating a subset of corpus '{}' for topic '{}'".format(corpus.fname, topic.topic_id))
        # specify the filter function
        def doc_filter(doc_dict: Dict[str, Any]) -> bool:
            """
            :return: True, iff this document has the specified topic
            """
            doc = documents[doc_dict['id']]
            return doc.topics and topic in doc.topics
        # build the new corpus
        corpus_path = self._get_corpus_path(topic)
        return corpus.subset(corpus_path, doc_filter)


    def test_model(self, fin_corpus: str, fin_model: str):
        model = self.load_model(fin_model)
        model.print_topics(num_topics=-1, num_words=10)

        corpus = JsonLinesCorpus(fin_corpus)
        for tokens in corpus:
            topics = model[tokens]
            print("dominant topics in https://arxiv.org/abs/{}".format(tokens))
            for topic, score in sorted(topics, key=lambda x: x[1], reverse=True):
                print("topic {} @ {:.3f}: {}".format(topic, score, model.print_topic(topic)))

    def test_document_topics(self):
        # get best matching documents + URLs per topic
        topic_model = DocumentCollection.from_dict(util.json_read(self.file_topics))

        docs_by_first_topic = defaultdict(list)
        # group documents by first topic
        for id, doc in topic_model.documents.items():
            if doc.topics:
                topic, score = doc.topics[0]
                docs_by_first_topic[topic].append((id, score))
            else:
                logger.warning("document {} has no topics".format(doc.doc_id))
        # sort by score descending
        for doc_list in docs_by_first_topic.values():
            doc_list.sort(key=lambda x: x[1], reverse=True)

        # print highest scoring documents for each topic
        for topic in topic_model.topics.values():
            print("Topic {}: {}".format(topic.topic_id, topic.tokens))
            for doc_id, score in docs_by_first_topic[topic.topic_id][:10]:
                print("paper https://arxiv.org/abs/{} with score {}".format(doc_id.replace('-', '/'), score))


    def docs_from_ids(self) -> Dict[str, Document]:
        return {doc_id: Document(doc_id) for doc_id in util.json_read(self.file_ids)}

    def docs_from_metadata(self, topics: List[Topic]) -> Dict[str, Document]:
        # restore documents
        topic_dict = {t.topic_id: t for t in topics}
        documents = Document.restore_documents(util.json_read(self.file_docs), topic_dict)
        # add topics to documents (one for each category)
        if self.category_layer:
            for doc in documents.values():
                if doc.categories:
                    for category in doc.categories:
                        doc.add_topic(topic_dict[category], 1.0)
                else:
                    logger.warning("Document {} has no categories!".format(doc.doc_id))
        return documents

    def topics_from_metadata(self, parent_topic: Topic) -> List[Topic]:
        # note: some papers do not have categories (especially very old ones)
        categories = (doc_dict['categories'] for doc_dict in util.json_read(self.file_docs) if doc_dict['categories'])
        topic_ids = sorted(set(util.flatten(categories, generator=True)))
        topics = [parent_topic.add_child(topic_id) for topic_id in topic_ids]
        return topics

    def load_dictionary(self) -> corpora.Dictionary:
        dictionary = corpora.Dictionary.load(self.file_dict)
        dictionary[0]  # forces id2token to be calculated. Probably a bug in gensim...
        return dictionary

    def load_corpus_for_topic(self, topic: Topic) -> JsonLinesCorpus:
        corpus_path = self._get_corpus_path(topic)
        if os.path.isfile(corpus_path):
            # load the corpus for this topic (if available)
            return JsonLinesCorpus(self._get_corpus_path(topic))
        else:
            if topic.parent:
                # ok, try again with this topic's parent
                return self.load_corpus_for_topic(topic.parent)
            else:
                # no parent left? then use the root corpus
                return JsonLinesCorpus(self.file_corpus)

    def _get_topic_file_prefix(self, topic: Topic) -> str:
        """
        get a file prefix based on the output path of this instance and the topic id
        """
        return "{}-topic-{}".format(self.file_output_prefix, topic.topic_id)

    def _get_model_path(self, topic: Topic) -> str:
        """
        get the path of the model associated with this topic
        """
        return self._get_topic_file_prefix(topic) + '-model.pkl.bz2'

    def _get_corpus_path(self, topic: Topic) -> str:
        """
        get the path of the model associated with this topic
        """
        return self._get_topic_file_prefix(topic) + '-corpus.json'

    @staticmethod
    def load_model(file_model: str) -> BaseTopicModel:
        logger.debug("loading model from file '{}'...".format(file_model))
        with util.open_by_ext(file_model, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def filter_by_lang(documents: Iterable[Dict[str, Any]], lang_code: str, threshold=0.8,
                       broken_codes=['cy', 'ca', 'pt']) -> Iterable[Dict[str, Any]]:
        logger.info("will only accept documents in language '{}'".format(lang_code))
        counter = Counter()
        for i, entry in enumerate(documents):
            id = entry['id']
            doc = entry['text']
            if not doc:
                logger.debug("empty document at index %s", i)
                continue
            sample = doc[5000:6000] if len(doc) >= 6000 else doc[:1000]
            try:
                langs = langdetect.detect_langs(sample)  # type: List[Language]
                lang = langs[0].lang
                proba = langs[0].prob
                if lang != lang_code or proba < threshold:
                    logger.debug("language: {}, {:.3f}, {}, \"{}\"".format(lang, proba, id, sample[:100].replace('\n', '\\n')))
                    if proba < threshold or lang in broken_codes:
                        counter['_failed'] += 1
                    else:
                        counter[lang] += 1
                else:
                    counter[lang] += 1
                    yield entry
            except LangDetectException:
                logger.warning("language detection failed on document {} (sample: {})".format(id, sample[:1000]), exc_info=1)
        logger.info("Results of language detection: {}".format(str(counter.most_common())))

    @classmethod
    def spacy_parse(cls, documents: Iterable[Dict[str, Any]], batch_size=10, n_threads=1) -> Iterable[Dict[str, Any]]:
        logger.debug("loading spacy model...")
        t = time.time()
        nlp = spacy.load('en', parser=False)
        logger.info("loading spacy model took {:.2f}s. Processing documents using spacy...".format(time.time() - t))

        # preserve document IDs
        gen1, gen2 = itertools.tee(documents)
        ids = (x['id'] for x in gen1)
        texts = (x['text'] for x in gen2)
        docs = nlp.pipe(texts)

        # start the actual work and join the results with the IDs again
        t = time.time()
        count = 0
        docs = nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads)
        for id, doc in zip(ids, docs): # type: Tuple[str, Doc]
            count += 1
            if count % batch_size == 0:
                t1 = time.time()
                logger.info("a total of {} documents has been processed, took {:.2f}s ({:.2f} doc/min, {} thread(s))"
                            .format(count, t1-t, batch_size*60/(t1-t), n_threads))
                t = t1
            # skip undesired tokens
            tokens = cls.filter_tokens(doc)
            lemmata = [token.lemma_ for token in tokens]
            yield {'id': id, 'tokens': lemmata}

    @staticmethod
    def filter_tokens(document: Doc) -> List[Token]:
        """
        conditions are
        - length > 1
        - first character is alpha
        - no space or punctuation
        - consists of few strange characters
        :param document:
        :return:
        """
        pos_filter = ['SPACE', 'PUNCT']
        return [token for token in document if
                len(token) > 1 and
                token.string[0].isalpha() and
                token.pos_ not in pos_filter and
                (token.is_alpha or text_processing.has_valid_chars(token.string))]

    def store_tokens_and_gensim_dict(self, documents: Iterable[Dict[str,Any]]):
        """
        process token stream to build dictionary in memory and dump tokens as one json per line to file.
        afterwards, serialize the entire dictionary.
        """
        logger.info("building the dictionary and storing the corpus...")
        dictionary = corpora.Dictionary()
        doc_ids = set()
        with util.open_by_ext(self.file_corpus_plain, 'wt', encoding='utf-8') as fp:
            for entry in documents:
                doc_id = entry['id']
                tokens = entry['tokens']    # type: List[str]
                token_counts = Counter(tokens)
                doc_ids.add(doc_id)
                result = {'id': doc_id, 'tokens': token_counts}
                fp.write(json.dumps(result, separators=None, indent=None, ensure_ascii=False))
                fp.write('\n')
                dictionary.doc2bow(tokens, allow_update=True)
        # store the document IDs
        util.json_write(sorted(doc_ids), self.file_ids)
        # store the dictionary
        dictionary.filter_extremes(no_below=self.token_min_count, no_above=0.2, keep_n=self.dict_size_limit)
        dictionary.compactify()
        dictionary.save(self.file_dict, pickle_protocol=4)
        return doc_ids, dictionary

    def store_gensim_dict(self, corpus: JsonLinesCorpus) -> Tuple[Set[str], corpora.Dictionary]:
        """
        process token stream to build dictionary in memory, then serialize the entire dictionary.
        also stores document IDs in a separate file.
        """
        logger.info("building the dictionary...")
        dictionary = corpora.Dictionary()
        doc_ids = set()
        for i, doc in enumerate(corpus.iter_all()):
            doc_id = doc['id']
            doc_ids.add(doc_id)
            token_counts = doc['tokens']    # type: Dict[str, int]
            # unfortunately, dictionary.doc2bow() does not accept (token,count) tuples
            # therefore we expand the dictionary to a token list again... (yes, this is stupid)
            tokens = util.flatten([token] * count for token, count in token_counts.items())
            dictionary.doc2bow(tokens, allow_update=True)
            if (i+1) % 50000 == 0:
                logger.info("{} documents have been read so far".format(i+1))
        # store the document IDs
        util.json_write(sorted(doc_ids), self.file_ids)
        # store the dictionary
        dictionary.filter_extremes(no_below=self.token_min_count, no_above=0.2, keep_n=self.dict_size_limit)
        dictionary.compactify()
        dictionary.save(self.file_dict, pickle_protocol=4)
        return doc_ids, dictionary

    @staticmethod
    def combine_pages(entry: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # document IDs might be broken, if they were extracted from file names...
        doc_id = text_processing.fix_file_based_id(entry['id'])
        raw_text = "\n".join(entry['pages'])
        clean_text = text_processing.clean_parsed_text(raw_text)
        return {'id': doc_id, 'text': clean_text}

    @staticmethod
    def get_title_and_abstract(entry: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        full_id = entry['header']['identifier']
        short_id = full_id[(full_id.rfind(':') + 1):]
        title = text_processing.strip_all_whitespace(entry['title'][0])
        abstract = text_processing.strip_all_whitespace(entry['description'][0])
        return {'id': short_id, 'text': (title + "\n\n" + abstract) }


def topic_stats(topic_file: str):
    print("gathering stats for topics in", topic_file)
    dc_dict = util.json_read(topic_file)
    dc = DocumentCollection.from_dict(dc_dict)
    flat_topics = util.flatten((doc.topics or [] for doc in dc.documents.values()), generator=True)
    c = Counter(flat_topics)
    for topic, count in c.most_common():
        print("{}: {} ({})".format(topic.topic_id, count, topic.tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a nested topic model and classify documents')
    parser.add_argument('-p', '--input-pdfs', metavar='FILE', type=str,
                        help='path to the file containing the parsed PDFs (output of pdf_parser)')
    parser.add_argument('-t', '--input-tokens', metavar='FILE', type=str,
                        help='path to the file containing the tokens of the parsed pdfs '
                             '(optional, alternative to --input-pdfs)')
    parser.add_argument('-m', '--input-meta', metavar='FILE', type=str,
                        help='path to the metadata file (output of arxiv_crawler. '
                             'required, if the category layer should be used)')
    parser.add_argument('-o', '--output-prefix', metavar='PATH', type=str, required=True,
                        help='all output files, including temporary files, will be prefixed '
                             'with this string. all results will be stored under this '
                             'prefix aswell.')
    parser.add_argument('-a', '--abstracts-only', action='store_true',
                        help="build topic models based on a paper's abstract only "
                             "(do not use the entire document text)")
    parser.add_argument('-T', '--topic-model', metavar='MODEL', type=str, default="hdp",
                        help='the topic model to use. Options: "hdp" (default), "lda")')
    parser.add_argument('-l', '--layers', metavar='LAYERS', type=str, default="10",
                        help='how many nested topic layers are to be used? Example: "10,7,4"')
    parser.add_argument('-c', '--limit-classification', metavar='LIMITS', type=str,
                        help='limits the number of topics that each document can be assigned '
                             'to at each layer during classification. One number per layer, '
                             '0 stands for unlimited. Must have same length as -l. '
                             'Example: "1,2,0"')
    parser.add_argument('-M', '--min-docs-per-topic', metavar='N', type=int,
                        help='require at least N documents per topic on each layer. '
                             'Can reduce the allowed topic count at each layer (but never increase). '
                             'Interrupts the build for a topic, if less than 3*N documents remain '
                             '(a topic model with just two topics does not seem useful)')
    parser.add_argument('-f', '--lang-filter', metavar='LANG', type=str, default="en",
                        help='filter by the specified language code. Defaults to "en" '
                             '(because we can currently only parse english text)')
    parser.add_argument('-v', '--vocab-size', metavar='N', type=int,
                        help='limit the size of the vocabulary, if specified')
    parser.add_argument('-d', '--doc-limit', metavar='N', type=int,
                        help='just process the first N documents (useful for testing)')
    args = parser.parse_args()
    # process list input & convert data types
    if isinstance(args.layers, str):
        args.layers = [int(s.strip()) for s in args.layers.split(",")]
    if isinstance(args.limit_classification, str):
        args.limit_classification = [int(s.strip()) for s in args.limit_classification.split(",")]
    if args.limit_classification and len(args.layers) != len(args.limit_classification):
        raise ValueError("the arguments --layers and --limit-classification must have the "
                         "same length! layers: {}, limits: {}"
                         .format(str(args.layers), str(args.limit_classification)))
    return args

# example args:
# topic_model.py -t "tokens.json.bz2" -m "meta.json.bz2" -o "./data/test" -l "5,5" -v 10000

if __name__ == "__main__":
    args = parse_args()
    topic_model = TopicModel(file_pdf_text=args.input_pdfs, file_corpus_input=args.input_tokens,
                             file_metadata=args.input_meta, file_output_prefix=args.output_prefix,
                             abstracts_only=args.abstracts_only, model=args.topic_model,
                             language_filter=args.lang_filter, batch_size=500, n_threads=None,
                             topic_layers=args.layers, topic_limit_per_layer=args.limit_classification,
                             category_layer=(args.input_meta is not None),
                             min_docs_per_topic=args.min_docs_per_topic,
                             token_min_count=5, dict_size_limit=args.vocab_size,
                             document_limit=args.doc_limit)
    topic_model.build()
