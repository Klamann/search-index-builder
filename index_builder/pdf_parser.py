import argparse
import logging
import os
import tarfile
from io import StringIO, BytesIO
from tarfile import TarFile, ExFileObject
from tarfile import TarInfo
from time import time
from typing import List, Dict, Iterable
from typing import Set

from pdfminer import settings
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError

import text_processing
import util

# fix pdfminer's logging issues
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('pdf-parser')
logger.setLevel(logging.INFO)

MB = 2**20
GB = 2**30


class PdfParser:

    def __init__(self, input_folder: str, output_folder: str, unpack_tar=False,
                 worker_threads=0, status_interval=0.1, parse_timeout=None, pdf_page_limit=0):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.unpack_tar = unpack_tar
        self.worker_threads = worker_threads
        self.status_interval = status_interval
        self.parse_timeout = parse_timeout
        self.pdf_page_limit = pdf_page_limit
        # the plaintext of the parsed pdfs
        self.file_plaintext_json = os.path.join(output_folder, 'plaintext.lines.json')
        # successfully parsed documents
        self.logfile_success = os.path.join(output_folder, 'parser-success.log')
        # documents that caused parser errors
        self.logfile_failure = os.path.join(output_folder, 'parser-failure.log')
        # documents that could not be parsed within the given time window
        self.logfile_timeout = os.path.join(output_folder, 'parser-timeout.log')
        # tar files that have been completely parsed
        self.logfile_tarfile = os.path.join(output_folder, 'parser-tarfile.log')
        # set representations of the logfile contents
        self.set_success = None     # type: Set[str]
        self.set_failure = None     # type: Set[str]
        self.set_timeout = None     # type: Set[str]
        self.set_tarfile = None     # type: Set[str]
        self.set_skip = None        # type: Set[str]
        self.entity_count = None

    def extract_pdf_texts(self):
        # create the output folder, if it does not yet exist
        os.makedirs(self.output_folder, exist_ok=True)
        # read logfiles and update data structures
        self.read_logfiles()
        # read pdfs from individual file or tar archives (as generator / stream)
        pdf_file_generator = self.pdfs_from_archives() if self.unpack_tar else self.pdfs_from_folder()
        # extract plain text from pdfs using pdfminer (as generator / stream)
        pdf_text_generator = self.parse_pdfs(pdf_file_generator)
        # write plain text to a file, with one json document per line (as generator / stream)
        util.json_write_lines(pdf_text_generator, self.file_plaintext_json,
                              append=(len(self.set_success) > 0), sort_keys=True)
        # because all of this was streamed, we only load a few files at a time in memory
        # and we can even parallelize all of this using all available cpu cores

    def read_logfiles(self):
        self.set_success = util.parse_task_log(self.logfile_success)
        self.set_failure = util.parse_task_log(self.logfile_failure)
        self.set_timeout = util.parse_task_log(self.logfile_timeout)
        self.set_tarfile = util.parse_task_log(self.logfile_tarfile)
        self.set_skip = set()
        self.set_skip.update(self.set_success, self.set_failure, self.set_timeout)
        if self.set_skip:
            logger.info("{} files were already processed and will be skipped".format(len(self.set_skip)))

    def pdfs_from_folder(self) -> Iterable['PdfFile']:
        """
        get a generator of all PDFs in a folder
        set self.entity_count
        :return: a generator containing all PDFs in the input folder (as bytes)
        """
        files = [file for file in sorted(os.listdir(self.input_folder)) if file.endswith(".pdf")]
        self.entity_count = len(files)
        for file in files:
            if self.is_file_to_skip(file):
                logger.debug("skipping {}".format(file))
            else:
                logger.debug("reading {}".format(file))
                yield PdfFile.from_file(os.path.join(self.input_folder, file))

    def pdfs_from_archives(self) -> Iterable['PdfFile']:
        # list files
        archives = [file for file in sorted(os.listdir(self.input_folder)) if file.endswith(".tar")]
        archives_skipped = [f for f in archives if f in self.set_tarfile]
        archives_remaining = [f for f in archives if f not in self.set_tarfile]

        # log stats
        # note: tar files have no index: to list their contents, they have to be read from start to end
        t0 = time()
        archive_size_total = sum(os.path.getsize(os.path.join(self.input_folder, file)) for file in archives_remaining)
        archive_size_processed = 0
        # separate stats for remaining time estimation
        est_size_total = archive_size_total
        est_size_processed = 0
        if archives_skipped:
            size_skipped = sum(os.path.getsize(os.path.join(self.input_folder, file)) for file in archives_skipped)
            logger.info("skipping {} archives (total size: {:.2f} GB) that are already done, continuing with {} archives (total size: {:.2f} GB)"
                        .format(len(archives_skipped), size_skipped/GB, len(archives_remaining), archive_size_total/GB))
        else:
            logger.info("processing {} archives (total size: {:.2f} GB)".format(len(archives_remaining), archive_size_total/GB))

        # read files from archives
        for i, archive in enumerate(archives_remaining):
            archive_path = os.path.join(self.input_folder, archive)
            archive_size = os.path.getsize(archive_path)
            logger.info("Opening archive {} ({:.1f} mb)".format(archive, os.path.getsize(archive_path)/MB))
            read = 0
            skipped = 0
            with tarfile.open(archive_path, "r") as tar:    # type: TarFile
                for tarinfo in tar:     # type: TarInfo
                    if tarinfo.isreg():
                        file_name = os.path.basename(tarinfo.name)
                        file_size = tarinfo.size
                        if self.is_file_to_skip(file_name):
                            skipped += 1
                            logger.debug("skipping {}".format(file_name))
                        else:
                            read += 1
                            logger.debug("reading {} ({:.2f} mb) from {}".format(file_name, file_size/MB, archive))
                            contents = tar.extractfile(tarinfo)     # type: ExFileObject
                            yield PdfFile(filename=file_name, binary=contents.read())
            # done with this archive, collect stats
            delta = time() - t0
            archive_size_processed += archive_size
            archive_size_read = (read / (read+skipped)) * archive_size
            est_size_processed += archive_size_read
            est_size_total -= (archive_size - archive_size_read)
            est_remaining_sec = (delta / max(est_size_processed, 1)) * (est_size_total - est_size_processed)
            logger.info("{}/{} archives ({:.1f} GB of {:.1f} GB, {:.2f} %) have been read in {:.2f} h (est. remaining: {:.2f} h)"
                        .format(i+1, len(archives_remaining), archive_size_processed/GB, archive_size_total/GB,
                                archive_size_processed * 100 / archive_size_total, delta / 3600,
                                est_remaining_sec / 3600))
            # write this archive to the task log
            util.write_task_log(archive, self.set_tarfile, self.logfile_tarfile)

    def is_file_to_skip(self, file: str) -> bool:
        return self.set_skip and self.filename_to_arxiv_id(file) in self.set_skip

    def parse_pdfs(self, pdf_files: Iterable['PdfFile']) -> Iterable[Dict]:
        """
        
        :param pdf_files: 
        :return: 
        """
        """
       Creates a generator that yields parsed pdfs.
       Can make use of a thread pool to speed up parsing.
       :param paths: the paths of the pdfs to parse
       :param parallel: uses all available threads, if set to True. Order is not guaranteed when parallel=True.
       :param status_interval: how often the status interval should be printed
              either every n documents or for values < 1 every n percent of documents.
              Default value 0.05 says print every 5% of all documents.
       :return: a generator containing the parsed PDFs. The generator is backed by a parallel Pool, so make sure
                to consume it fast or documents will pile up in memory.
       """
        t0 = time()
        num_threads = self.worker_threads if self.worker_threads else None
        task_size = None    # need to fill in later, due to lazy evaluation

        # determine the absolute status interval
        relative = self.status_interval < 1 and self.entity_count
        status_interval = 100
        if relative:
            status_interval = int(self.entity_count / (1.0 / self.status_interval))
        elif self.status_interval >= 1:
            status_interval = self.status_interval
        else:
            logger.warning("The status interval was set to every {:.1} %, but we don't know "
                           "the total number of PDFs to parse. As a fallback, we'll print a "
                           "status update every 100 files".format(self.status_interval / 100))

        # initialize counters
        counter = 0
        failures = 0
        sum_pages = 0

        logger.info('preparing to parse pdfs using {} thread{}'
                    .format(num_threads or os.cpu_count(), "" if num_threads == 1 else "s"))
        try:
            pool = util.ProcessKillingExecutor(max_workers=num_threads)

            def timeout_function(pdf_file: PdfFile):
                short_id = os.path.splitext(pdf_file.filename)[0]
                logger.warning("parsing of {} was interrupted after a timeout of {}s".format(short_id, self.parse_timeout))
                util.write_task_log(short_id, self.set_timeout, self.logfile_timeout)

            # only parse pdfs that we didn't already parse before
            filtered_pdf_stream = (pdf for pdf in pdf_files if self.filename_to_arxiv_id(pdf.filename) not in self.set_skip)
            # hand the generator over to the process pool
            results = pool.map(self.parse_pdf, filtered_pdf_stream, timeout=self.parse_timeout, callback_timeout=timeout_function)

            for result in results:
                counter += 1
                if result and 'failure' not in result:
                    sum_pages += len(result['pages'])
                    if counter % status_interval == 0:
                        delta = time() - t0
                        stats = "{:.2f} pdfs per minute, {:.2f} pages per second".format((counter / delta) * 60, sum_pages / delta)
                        if self.entity_count:
                            if not task_size:
                                task_size = (self.entity_count - len(self.set_skip))
                            logger.info("{} of {} pdfs ({:.2f} %) have been parsed in {:.1f} min ({}, est. remaining: {:.1f} min)"
                                        .format(counter, task_size, counter * 100 / task_size, delta / 60, stats, (delta/60) * (task_size - counter) / counter))
                        else:
                            logger.info("{} pdfs have been parsed in {:.1f} min ({})".format(counter, delta / 60, stats))
                    yield result
                    util.write_task_log(result['id'], self.set_success, self.logfile_success)
                else:
                    failures += 1
                    if result:
                        util.write_task_log(result['id'], self.set_failure, self.logfile_failure)
        finally:
            time_all = time() - t0
            logger.info("finished parsing of {} pdfs in {:.2f}s ({:.2f} pdfs per minute, {:.2f} pages per second)."
                        " {} documents have been skipped due to parsing errors."
                        .format(counter, time_all, (counter / time_all) * 60, sum_pages / time_all, failures))

    def parse_pdf(self, pdf_file: 'PdfFile') -> Dict:
        t0 = time()
        short_id = self.filename_to_arxiv_id(pdf_file.filename)
        if len(pdf_file.binary) == 0:
            logger.warning("failed to parse `{}`: file is empty".format(pdf_file.filename))
            return {'id': short_id, 'failure': "empty file"}
        try:
            pages = self.pdf_extract_text(pdf_file, maxpages=self.pdf_page_limit)
            logger.debug("finished parsing {} ({:5.2f} mb, {:>3} pages) after {:5.2f}s"
                        .format(pdf_file.filename, pdf_file.binary.__sizeof__() / MB, len(pages), time() - t0))
            return {'id': short_id, 'pages': pages}
        except PDFSyntaxError as e:
            logger.warning("failed to parse `{}`: {}".format(pdf_file.filename, str(e)))
            return {'id': short_id, 'failure': str(e)}
        except Exception as e:
            logger.warning("failed to parse `{}`".format(pdf_file.filename), exc_info=1)
            return {'id': short_id, 'failure': str(e)}

    @classmethod
    def pdf_extract_text(cls, pdf_file: 'PdfFile', maxpages=0) -> List[str]:
        """

        :param pdf_file:
        :return: the pages of the pdf, one string per page
        """
        # TODO mabe pass return object & add break condition (requires adjustments in executor...)
        retstr = StringIO()
        settings.STRICT = False     # try to skip errors whenever possible
        rsrcmgr = PDFResourceManager(caching=False)
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        text_pages = []
        last_index = 0
        fp = BytesIO(pdf_file.binary)
        for page in PDFPage.get_pages(fp, maxpages=maxpages):
            interpreter.process_page(page)
            page_text = retstr.getvalue()[last_index:]
            text_pages.append(page_text)
            last_index += len(page_text)
        device.close()
        retstr.close()
        return text_pages

    @classmethod
    def filename_to_arxiv_id(cls, filename):
        # need to fix broken IDs for documents extracted from the official tar archives
        short_id = os.path.basename(filename[:-4].replace('_', '/'))
        return text_processing.fix_file_based_id(short_id)

    @classmethod
    def test_print_pdf(cls, file):
        print("reading ", file)
        t0 = time()
        pdf_file = PdfFile.from_file(file)
        pages = cls.pdf_extract_text(pdf_file)
        print("parsing took {:.2f}s".format(time() - t0))
        for page in pages:
            print(page)


class PdfFile:

    def __init__(self, filename: str, binary: bytes):
        super().__init__()
        self.filename = filename
        self.binary = binary

    @staticmethod
    def from_file(file):
        return PdfFile(filename=os.path.basename(file), binary=file_to_bytes(file))

    def __str__(self):
        return "PdfFile: {}, {:.2f} MB".format(self.filename, len(self.binary) / MB)

    def __repr__(self):
        return self.__str__()


def file_to_bytes(path: str):
    with open(path, "rb") as fp:
        return fp.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse PDFs and store them as plaintext documents')
    parser.add_argument('-i', '--input', metavar='FOLDER', type=str, required=True,
                        help='path to the input folder, where the PDFs are stored')
    parser.add_argument('-o', '--output', metavar='FOLDER', type=str, required=True,
                        help='path to the output folder, where the parsed plain text and logfiles will be stored')
    parser.add_argument('--tar', action='store_true', help='read PDFs from tar archives in the input folder')
    parser.add_argument('-p', '--pagelimit', metavar='N', type=int, default=0,
                        help='limit the number of pages to parse per pdf (default: unlimited)')
    parser.add_argument('-t', '--timeout', metavar='SECONDS', type=int, default=60,
                        help='parse timeout in seconds for a single pdf (default: 60 seconds)')
    parser.add_argument('-w', '--workers', metavar='N', type=int, default=0,
                        help='the number of parallel worker threads to use for pdf parsing '
                             '(default: 0, which means use all available cpu cores)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parser = PdfParser(args.input, args.output, unpack_tar=args.tar, worker_threads=args.workers,
                       status_interval=100, parse_timeout=args.timeout, pdf_page_limit=args.pagelimit)
    parser.extract_pdf_texts()
