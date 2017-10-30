import argparse
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime
from typing import Dict

import requests
from sickle import Sickle
from sickle.iterator import OAIItemIterator
from sickle.models import Record

import util

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('arxiv-crawler')
logger.setLevel(logging.DEBUG)


# arxiv OAI acces documentation:
# https://arxiv.org/help/oa/index

MB = 2**20
URL_OAI2 = 'http://export.arxiv.org/oai2'
URL_PDF = 'https://arxiv.org/pdf/'


def build_oai_params(set: str = None, date_from: str = None, date_until: str = None) -> Dict[str, str]:
    oai_params = {
        'metadataPrefix': 'oai_dc',
    }
    if set:
        oai_params['set'] = set
    if date_from:
        oai_params['from'] = date_from
    if date_until:
        oai_params['until'] = date_until
    return oai_params


def crawl_metadata(output_file, oai_params=None, fetch_limit=None):
    """
    crawls records, flushes them regularily to a temporary json file.
    low memory footprint, no loss of intermediate results.
    """
    sickle = Sickle(URL_OAI2)
    oai_params = oai_params if oai_params else {}

    logger.info("{} - retrieving records from {} with params {}".format(str(datetime.now()), URL_OAI2, str(oai_params)))
    t0 = time.time()
    t_last = t0
    raw_records = sickle.ListRecords(**oai_params)  # type: OAIItemIterator

    metadata_list = []
    records_size = int(raw_records._get_resumption_token().complete_list_size)
    batch_counter = 0
    batch_size = raw_records.oai_response.http_response.content.decode().count("</record>")
    batch_sum = int(math.ceil(records_size / float(batch_size)))
    counter = 0
    for raw_record in raw_records:  # type: Record
        # parse element and append
        try:
            identifier, record = parse_raw_record(raw_record)
            if record:
                metadata_list.append(record)
            else:
                logger.debug("Record `{}` was deleted and will therefore not appear in the results.".format(identifier))
        except Exception:
            logger.warning("Failed to parse record %s", str(raw_record), exc_info=1)

        # write batch to file and write log
        counter += 1
        if counter % batch_size == 0:
            # write batch to file
            batch_counter += 1
            util.json_write_lines(metadata_list, output_file, append=(batch_counter > 1))
            metadata_list = []

            # log event
            t_current = time.time()
            t_remaining = ((1 / (counter / records_size)) - 1) * (t_current - t0)
            logger.info("Batch {}/{}: fetched {} of {} records (took {}s, remaining: {} min, resumption token: {})".format(
                batch_counter, batch_sum, counter, records_size, round(t_current - t_last, 2), round(t_remaining / 60, 1),
                raw_records._get_resumption_token().token))
            t_last = t_current
        if fetch_limit and counter >= fetch_limit:
            break

    # write last batch
    if len(metadata_list) > 0:
        logger.info("Batch {}/{}: fetched the remaining {} records".format(batch_sum, batch_sum, len(metadata_list)))
        util.json_write_lines(metadata_list, output_file, append=True)

    logger.info("All {} entries were retrieved in {}s and written to {}".format(counter, round(time.time() - t0), output_file))


def crawl_pdfs(metadata_file, job_folder='./crawl', sleep_time=5, size_limit_mb=10):
    logger.warning("This function crawls pdf documents from arxiv.org, which is "
                   "strongly discouraged as it puts high load on the servers. "
                   "Please use it only to retrieve small portions of the dataset "
                   "(no more than 1k documents).\n"
                   "Please download the PDFs from Amazon S3 instead. "
                   "Getting the entire dataset costs about $70 as of 2017.")
    t0 = time.time()
    size_limit = size_limit_mb * MB
    # define folders
    pdf_folder = os.path.join(job_folder, "pdf")
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    log_success = os.path.join(job_folder, "crawl-done.log")
    log_skipped = os.path.join(job_folder, "crawl-skipped.log")
    log_failure = os.path.join(job_folder, "crawl-failed.log")
    # parse logs if existing
    set_success = util.parse_task_log(log_success)
    set_skipped = util.parse_task_log(log_skipped)
    set_failure = util.parse_task_log(log_failure)
    # start the loop
    headers = {'ACCEPT_ENCODING': 'gzip, deflate, br', 'USER_AGENT': 'Mozilla/5.0'}
    counter = 0
    size = sum(1 for line in util.json_read_lines(metadata_file))
    skipped = 0
    for record in util.json_read_lines(metadata_file):
        counter += 1
        identifier = record['header']['identifier']
        arxiv_id = identifier[(identifier.rfind(':') + 1):]
        try:
            title = record.get('title', '')[0]
            pdf_file = os.path.join(pdf_folder, arxiv_id.replace('/', '_') + '.pdf')
            if arxiv_id in set_skipped or (arxiv_id in set_success):
                skipped += 1
            else:
                if skipped > 0:
                    logger.info("Skipped {} already existing entries".format(skipped))
                    skipped = 0
                url_generated = URL_PDF + arxiv_id
                url = requests.head(url_generated, headers=headers).headers.get('location')
                file_size = int(requests.head(url, headers=headers).headers.get('content-length', 0))
                if file_size == 0:
                    logger.warn("Document {} at {} is empty!".format(arxiv_id, url))
                    util.write_task_log(arxiv_id, set_skipped, log_skipped)
                    time.sleep(1)
                elif file_size > size_limit:
                    logger.warn("Document {} '{}' at {} has {:.2f} mb (exceeds the size limit of {:.2f} mb). Skipping..."
                                .format(arxiv_id, title[:30], url, file_size/MB, size_limit/MB))
                    util.write_task_log(arxiv_id, set_skipped, log_skipped)
                    time.sleep(1)
                else:
                    print("{} - {}/{}: requesting arXiv:{} ({:.2f} kb) '{}'... "
                          .format(datetime.now().isoformat(), counter, size, arxiv_id, file_size / 1024, title[:30]), end='')
                    r = requests.get(url, stream=True, headers=headers)
                    with open(pdf_file, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                        util.write_task_log(arxiv_id, set_success, log_success)
                    print('done. sleeping for about {}s'.format(sleep_time))
                    sleep_time_rand = sleep_time * 0.7 + sleep_time * 0.6 * random.uniform(0, 1)
                    time.sleep(sleep_time_rand)
        except Exception as e:
            logging.exception("Failed to retrieve {}".format(identifier))
            util.write_task_log(arxiv_id, set_failure, log_failure)
            time.sleep(1)


def parse_raw_record(raw_record: Record):
    identifier = raw_record.header.identifier
    record = None
    if not raw_record.deleted:
        record = raw_record.metadata
        record['header'] = {
            'datestamp': raw_record.header.datestamp,
            'deleted': raw_record.header.deleted,
            'identifier': identifier,
            'setSpecs': raw_record.header.setSpecs,
        }
    return identifier, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Retrieve metadata and documents from arxiv.org')
    subparsers = parser.add_subparsers(title='actions', dest='action')

    # metadata download
    help_meta = "download metadata via the official OAI endpoint on arxiv.org. " \
                "This is the canonical way to retrieve metadata for all documents " \
                "that have been submitted to arxiv."
    parser_meta = subparsers.add_parser('meta', help=help_meta)
    parser_meta.set_defaults(func=action_meta)
    parser_meta.add_argument('-o', '--output', metavar='FILE', type=str, required=True,
                             help='path to the file where the metadata will be stored. '
                                  'Use `.gz` or `.bz2` as file extension to enable compression')
    parser_meta.add_argument('-s', '--set', metavar='SET', type=str,
                             help='restrict downloaded data to documents from this set. List of '
                                  'available options: http://export.arxiv.org/oai2?verb=ListSets')
    parser_meta.add_argument('-f', '--dateFrom', metavar='DATE', type=str,
                             help='only retrieve documents from this date or later. '
                                  'Specify the date in ISO format, e.g. "2017-01-31"')
    parser_meta.add_argument('-u', '--dateUntil', metavar='DATE', type=str,
                             help='only retrieve documents until this date (inclusive). '
                                  'Specify the date in ISO format, e.g. "2017-01-31"')

    # pdf download
    help_pdfs = "download pdfs from the arxiv website. WARNING: please don't use this method " \
                "for bulk downloads of more than a few files! If you want to get a large " \
                "portion of the arxiv dataset, please read https://arxiv.org/help/bulk_data"
    parser_pdfs = subparsers.add_parser('pdfs', help=help_pdfs)
    parser_pdfs.set_defaults(func=action_pdfs)
    parser_pdfs.add_argument('-i', '--input', metavar='FILE', type=str, required=True,
                             help='path to the metadata file that contains the identifiers of '
                                  'the documents to retrieve. Hint: Get the metadata file '
                                  'using the `meta` command')
    parser_pdfs.add_argument('-o', '--output', metavar='FOLDER', type=str, required=True,
                             help='path to the output folder, where the downloaded pdfs '
                                  'and logfiles will be stored')
    # parse args, print instructions if necessary
    args = parser.parse_args()
    if not args.action:
        parser.print_help()
    return args


def action_meta(args):
    oai_params = build_oai_params(set=args.set, date_from=args.dateFrom, date_until=args.dateUntil)
    crawl_metadata(args.output, oai_params)


def action_pdfs(args):
    crawl_pdfs(args.input, job_folder=args.output)


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
