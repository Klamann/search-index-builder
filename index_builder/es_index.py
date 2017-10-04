import logging
from typing import Dict, Any

from elasticsearch import Elasticsearch

import text_processing
import util
from categories import Acm98
from data import DocumentCollection, Document

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('es-index')
logger.setLevel(logging.DEBUG)

local = True
host = 'localhost' if local else 'es2.nx42.de'
port = 9200
node = {'host': host, 'port': port}
with open('res/arxiv-mapping.json', 'r') as fp:
    mapping_arxiv = fp.read()
es = Elasticsearch([node], http_auth=(None if local else ('admin', 'pinkie')))
print(es.info())

# launch config
index_new = False
index_overwrite = False
index_name = "arxiv-index"
file_meta = None    #'../data/arxiv-meta-full-lines.json'
file_topics = None  # '../data/arxiv-topics.json.bz2'
file_content = None # '../data/pdf-text-cs-all-sorted.json'

# updates
add_mapping_url = 'arxiv-index/_mapping/arxiv'
add_mapping_put = {"properties": {"acm": {"type": "string", "index": "not_analyzed"}}}


def main(index_new=True, index_overwrite=False, file_meta: str = None, file_topics: str = None, file_content: str = None):
    if index_new:
        logger.info("creating new index on %s" % host)
        create_index(overwrite=index_overwrite)
    if file_meta:
        logger.info("populating index with metadata from %s" % file_meta)
        update_metadata(file_meta, chunk_size=10000)
    if file_topics:
        logger.info("updating documents with topic information from %s" % file_topics)
        update_topics(file_topics, chunk_size=10000)
    if file_content:
        logger.info("updating documents with text contents from %s" % file_content)
        update_content(file_content, chunk_size=1000)


def create_index(overwrite=True):
    if overwrite and es.indices.exists(index=index_name):
        deleted = es.indices.delete(index=index_name)
        logger.info("existing index has been overwritten: %s" % deleted)
    created = es.indices.create(index=index_name, body=mapping_arxiv)
    logger.info("new index has been created: %s" % created)


def update_metadata(file, chunk_size=1000):
    actions = util.json_read_lines(file, record_create)
    util.es_bulk(es, actions, chunk_size=chunk_size)


def update_content(file, chunk_size=1000):
    actions = util.json_read_lines(file, pdf_update)
    util.es_bulk(es, actions, chunk_size=chunk_size)


def update_topics(file, chunk_size=1000):
    dc_dict = util.json_read(file)
    dc = DocumentCollection.from_dict(dc_dict)
    logger.info("finished parsing input. updating {} documents with {} different topics..."
                .format(len(dc.documents), len(dc.topics)))
    actions = (topics_update(doc) for doc in dc.documents.values())
    util.es_bulk(es, actions, chunk_size=chunk_size)


def record_create(record: Dict[str, Any], index='arxiv-index', type='arxiv', acm: Acm98 = None):
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


def pdf_update(entry, index='arxiv-index', type='arxiv'):
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


def topics_update(doc: Document, index='arxiv-index', type='arxiv'):
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


if __name__ == "__main__":
    main(index_new=index_new, index_overwrite=index_overwrite, file_meta=file_meta,
         file_topics=file_topics, file_content=file_content)
