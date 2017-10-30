import argparse
import logging
from typing import Dict, Any

from elasticsearch import Elasticsearch

import text_processing
import util
from data import DocumentCollection, Document

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('es-index')
logger.setLevel(logging.DEBUG)


def main(es_host='localhost', es_port=9200, http_user=None, http_password=None,
         index_name='arxiv-index', index_type='arxiv', index_new=True, index_overwrite=False,
         file_meta: str = None, file_topics: str = None, file_content: str = None):
    """
    Creates and populates an Elasticsearch index using the specified data sources.
    :param es_host: the network host of the Elasticsearch node (default: localhost)
    :param es_port: the tcp port Elasticsearch is listening on (default: 9200)
    :param http_user: if http authentication is required, please specify the name here
    :param http_password: if http authentication is required, please specify the password here
    :param index_name: the name of the Elasticsearch index to use
    :param index_type: the type name of the Elasticsearch index to use
    :param index_new: when set to True, create a new index (otherwise we assume an index already exists)
    :param index_overwrite: when set to True, overwrite an index of the same name, if it already
           exists (otherwise we abort)
    :param file_meta: path to the metadata file (see `arxiv_crawler.py` to get metadata).
           Can read bz2 and gzip compressed files.
    :param file_topics: path to the topic model file (see `topic_model.py` to build a topic model).
           Can read bz2 and gzip compressed files.
    :param file_content: path to the file containing the PDF texts (see `pdf_parser.py` to extract
           text from PDFs). Can read bz2 and gzip compressed files.
    """
    if not(index_new or file_meta or file_topics or file_content):
        logger.info("No relevant parameters have been provided, terminating.")
        return
    es = connect_es(es_host, es_port, http_user=http_user, http_password=http_password)
    if index_new:
        logger.info("creating new index on %s" % es_host)
        create_index(es, index_name, overwrite=index_overwrite)
    if file_meta:
        logger.info("populating index with metadata from %s" % file_meta)
        update_metadata(es, file_meta, chunk_size=10000, index_name=index_name, index_type=index_type)
    if file_topics:
        logger.info("updating documents with topic information from %s" % file_topics)
        update_topics(es, file_topics, chunk_size=10000, index_name=index_name, index_type=index_type)
    if file_content:
        logger.info("updating documents with text contents from %s" % file_content)
        update_content(es, file_content, chunk_size=1000, index_name=index_name, index_type=index_type)
    logger.info("all actions completed")


def connect_es(host='localhost', port=9200, http_user=None, http_password=None,
               log_info=True) -> Elasticsearch:
    """
    Create a new connection to the specified Elasticsearch node
    :param host: the network host of the node
    :param port: the tcp port Elasticsearch is listening on (by default: 9200)
    :param http_user: if http authentication is required, use this name
    :param http_password: if http authentication is required, use this password
    :param log_info: log some diagnostic information about the cluster when connecting
    :return: the Elasticsearch low-level client
    """
    node = {'host': host, 'port': port}
    http_auth = (http_user, http_password) if http_user else None
    es = Elasticsearch([node], http_auth=http_auth)
    if log_info:
        logger.info(es.info())
    return es


def create_index(es: Elasticsearch, index_name: str, overwrite=True):
    """
    Create a new Elasticsearch index
    :param es: an Elasticsearch client
    :param index_name: the name of the index to create
    :param overwrite: set to True to overwrite an existing index (raises an exception otherwise,
           if an index of the same name already exists)
    """
    with open('res/arxiv-mapping.json', 'r') as fp:
        mapping_arxiv = fp.read()
    if overwrite and es.indices.exists(index=index_name):
        deleted = es.indices.delete(index=index_name)
        logger.info("existing index has been overwritten: %s" % deleted)
    created = es.indices.create(index=index_name, body=mapping_arxiv)
    logger.info("new index has been created: %s" % created)


def update_metadata(es: Elasticsearch, file: str, index_name='arxiv-index', index_type='arxiv',
                    chunk_size=1000):
    """
    Update an index with the metadata from the specified file.
    Use `arxiv_crawler.py` to get metadata. Please refer to `README.md` for further instructions.
    :param es: an Elasticsearch client
    :param file: the metadata file, as json (files with gzip or bz2 compression are accepted too)
    :param index_name: the name of the index to update
    :param index_type: the type of the index to update
    :param chunk_size: creates bulk requests with the specified chunk siye
           (a high chunk size may decrease indexing time, but increases memory usage)
    """
    actions = util.json_read_lines(file, record_create, index=index_name, type=index_type)
    util.es_bulk(es, actions, chunk_size=chunk_size)


def update_content(es: Elasticsearch, file: str, index_name='arxiv-index', index_type='arxiv',
                   chunk_size=1000):
    """
    Update an index with the plaintext representation of the parsed PDFs from the specified file.
    Use `pdf_parser.py` to extract plaintext from PDFs. Please refer to `README.md` for further
    instructions.
    :param es: an Elasticsearch client
    :param file: the file with the pdf contents, as json (files with gzip or bz2 compression are
           accepted too)
    :param index_name: the name of the index to update
    :param index_type: the type of the index to update
    :param chunk_size: creates bulk requests with the specified chunk siye
           (a high chunk size may decrease indexing time, but increases memory usage)
    """
    actions = util.json_read_lines(file, pdf_update, index=index_name, type=index_type)
    util.es_bulk(es, actions, chunk_size=chunk_size)


def update_topics(es: Elasticsearch, file: str, index_name='arxiv-index', index_type='arxiv',
                  chunk_size=1000):
    """
    Update an index with the topic model from the specified file.
    Use `topic_model.py` to generate a topic model. Please refer to `README.md` for further
    instructions.
    :param es: an Elasticsearch client
    :param file: the file with the pdf contents, as json (files with gzip or bz2 compression are
           accepted too)
    :param index_name: the name of the index to update
    :param index_type: the type of the index to update
    :param chunk_size: creates bulk requests with the specified chunk siye
           (a high chunk size may decrease indexing time, but increases memory usage)
    """
    dc_dict = util.json_read(file)
    dc = DocumentCollection.from_dict(dc_dict)
    logger.info("finished parsing input. updating {} documents with {} different topics..."
                .format(len(dc.documents), len(dc.topics)))
    actions = (topics_update(doc, index=index_name, type=index_type) for doc in dc.documents.values())
    util.es_bulk(es, actions, chunk_size=chunk_size)


def record_create(record: Dict[str, Any], index='arxiv-index', type='arxiv') -> Dict[str, Any]:
    """
    Build an Elasticsearch action that creates a new document from the specified metadata record.
    :param record: the metadata record as dict (result of `arxiv_crawler.py`)
    :param index: the name of the index to create the record in
    :param type: the type of the index to create the record in
    :return: an Elasticsearch action as dict
    """
    header = record['header']
    full_id = header['identifier']
    short_id = full_id[(full_id.rfind(':') + 1):]
    abstract = text_processing.strip_all_whitespace(record['description'][0])
    title = text_processing.strip_all_whitespace(record['title'][0])
    subjects = record['subject']
    return {
        "_op_type": "create",
        "_index": index,
        "_type": type,
        "_id": short_id,
        "_source": {
            "arxiv-id": short_id,
            "arxiv-url": record['identifier'][0],
            "title": title,
            "authors": record['creator'],
            "date-created": record['date'][0],
            "date-submitted": header['datestamp'],
            "categories": header['setSpecs'],
            "subjects": subjects,
            "abstract": abstract,
        }
    }


def pdf_update(entry: Dict[str, Any], index='arxiv-index', type='arxiv') -> Dict[str, Any]:
    """
    Build an Elasticsearch action that updates an existing document with the specified text.
    :param entry: the plain text entry as dict (result of `pdf_parser.py`)
    :param index: the name of the index to create the record in
    :param type: the type of the index to create the record in
    :return: an Elasticsearch action as dict
    """
    short_id = text_processing.fix_file_based_id(entry['id'])
    pages = entry['pages']
    pages_clean = [text_processing.clean_parsed_text(page) for page in pages]
    return {
        "_op_type": "update",
        "_index": index,
        "_type": type,
        "_id": short_id,
        "doc": {
            "pages": pages_clean
        }
    }


def topics_update(doc: Document, index='arxiv-index', type='arxiv') -> Dict[str, Any]:
    """
    Build an Elasticsearch action that updates an existing document with the specified topics.
    :param doc: the document that contains the topics (result of `topic_model.py`)
    :param index: the name of the index to create the record in
    :param type: the type of the index to create the record in
    :return: an Elasticsearch action as dict
    """
    short_id = text_processing.fix_file_based_id(doc.doc_id)
    nested_type_topic = [{
        'topic': t.topic_id,
        'score': score,
        'layer': t.layer
    } for t, score in (doc.topics.items() if doc.topics else [])]
    return {
        "_op_type": "update",
        "_index": index,
        "_type": type,
        "_id": short_id,
        "doc": {
            "topics": nested_type_topic
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and populate an Elasticsearch index.")
    parser.add_argument('-H', '--host', metavar='HOST', type=str, default="localhost",
                        help="the network host of the Elasticsearch node (default: localhost)")
    parser.add_argument('-P', '--port', metavar='PORT', type=int, default=9200,
                        help="the tcp port Elasticsearch is listening on (default: 9200)")
    parser.add_argument('-u', '--http-user', metavar='USER', type=str,
                        help="if http authentication is required, please specify the name here")
    parser.add_argument('-p', '--http-password', metavar='PASS', type=str,
                        help="if http authentication is required, please specify the password here")
    parser.add_argument('-i', '--index-name', metavar='NAME', type=str, default="arxiv-index",
                        help="the name of the Elasticsearch index to use")
    parser.add_argument('-T', '--index-type', metavar='NAME', type=str, default="arxiv",
                        help="the type name of the Elasticsearch index to use")
    parser.add_argument('-n', '--new-index', action='store_true',
                        help="create a new index (otherwise we assume an index already exists)")
    parser.add_argument('-o', '--overwrite-index', action='store_true',
                        help="overwrite an index of the same name, if it already exists "
                             "(otherwise we abort)")
    parser.add_argument('-m', '--file-meta', metavar='FILE', type=str,
                        help="path to the metadata file (see `arxiv_crawler.py` to get metadata). "
                             "Can read bz2 and gzip compressed files.")
    parser.add_argument('-t', '--file-topics', metavar='FILE', type=str,
                        help="path to the topic model file (see `topic_model.py` to build a "
                             "topic model). Can read bz2 and gzip compressed files.")
    parser.add_argument('-c', '--file-content', metavar='FILE', type=str,
                        help="path to the file containing the PDF texts (see `pdf_parser.py` "
                             "to extract text from PDFs). Can read bz2 and gzip compressed files.")
    return parser.parse_args()


# example args:
# es_index.py --host "127.0.0.1" --port 9200 --new-index --index-name "test-index" --file-meta /path/to/meta.json.bz2


if __name__ == "__main__":
    args = parse_args()
    main(es_host=args.host, es_port=args.port, http_user=args.http_user,
         http_password=args.http_password, index_name=args.index_name, index_type=args.index_type,
         index_new=args.new_index, index_overwrite=args.overwrite_index, file_meta=args.file_meta,
         file_topics=args.file_topics, file_content=args.file_content)
