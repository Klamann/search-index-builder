import bz2
import collections
import datetime
import gzip
import itertools
import json
import logging
import os
import time
from io import UnsupportedOperation
from multiprocessing import Manager, Process
from threading import Thread, Lock, Condition
from typing import Callable, Dict, Any, Iterable, List, Generator, IO, T, Union, Set

from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger('index-builder')
logger.setLevel(logging.INFO)


def json_write(data, file, pretty=True, sort_keys=True, append=False):
    indent = 2 if pretty else None
    with open_by_ext(file, 'at' if append else 'wt', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def json_read(file):
    with open_by_ext(file, 'rt', encoding='utf-8') as fp:
        return json.load(fp)


def json_write_lines(iterable, file, append=False, sort_keys=False):
    with open_by_ext(file, 'at' if append else 'wt', encoding='utf-8') as fp:
        for elem in iterable:
            fp.write(json.dumps(elem, sort_keys=sort_keys, indent=None, separators=None, ensure_ascii=False))
            fp.write('\n')
            fp.flush()


def json_read_lines(file: str, function_yielding: Callable[[Dict], T] = None, **kwargs) -> Generator[T, None, None]:
    with open_by_ext(file, 'rt', encoding='utf-8') as fp:
        for line in fp:
            record = json.loads(line)
            yield function_yielding(record, **kwargs) if function_yielding else record


def open_by_ext(filename, mode='r', **kwargs) -> IO:
    if filename.endswith('.bz2'):
        return bz2.open(filename, mode=mode, **kwargs)
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, **kwargs)
    else:
        return open(filename, mode=mode, **kwargs)


def is_stream_seekable(stream: IO) -> bool:
    try:
        stream.tell()
        return True
    except UnsupportedOperation:
        return False



def grouper(chunk_size: int, iterable: Iterable[T]) -> Generator[List[T], None, None]:
    it = iter(iterable)
    while True:
       chunk = list(itertools.islice(it, chunk_size))
       if not chunk:
           return
       yield chunk


def truth_filter(itr: Iterable) -> List:
    """
    creates a new list containing all elements of this iterable that are true
    (i.e. do not evaluate to False in an if statement, like None, 0 and False)
    :param itr: some iterable
    :return: a list containing all true elements of the specified iterable
    """
    return [x for x in itr if x]


def flatten(iterable: Iterable[Iterable[T]], generator=False) -> Union[List[T], Iterable[T]]:
    """flattens a sequence of nested elements"""
    flat = itertools.chain.from_iterable(iterable)
    return flat if generator else list(flat)


def es_bulk(es: Elasticsearch, actions: Iterable, chunk_size=1000, request_timeout=600):
    count_bulks = 0
    count_docs = 0
    failure_agg = []
    for chunk in grouper(chunk_size, actions):
        t0 = time.time()
        count_bulks += 1
        size = len(str(chunk))
        num_success, failures = helpers.bulk(
            es, chunk, chunk_size=chunk_size, request_timeout=request_timeout, raise_on_error=False)
        count_docs += num_success
        delta = time.time() - t0
        logger.info('{} actions (~{:.2f} mb) have been executed in bulk {} '
                    '(took {:.2f}s, {:.1f} actions/minute)'
                    .format(len(chunk), size / 2 ** 20, count_bulks, delta, 60 * len(chunk)/delta))
        if failures:
            failure_agg.extend(failures)
            logger.warning("{} failures in last bulk: {}".format(len(failures), failures))
    logger.info("added {} documents in {} bulks of size {} to the index, there were {} failures"
                .format(count_docs, count_bulks, chunk_size, len(failure_agg)))


def parse_task_log(logfile: str):
    """
    parses an existing log and returns the content of each line in a new set (duplicate-free)
    :param logfile: the file to parse
    :return: a set containing each line of the file
    """
    entries = set()
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            for line in f:
                entries.add(line.strip())
    return entries


def write_task_log(entry: str, entry_set: set, logfile: str):
    """
    writes a new task log, by appending entries to a file
    :param entry: the entry to add
    :param entry_set: a set of all already existing entries (we do not want to write duplicates)
    :param logfile: the file to write to
    """
    if entry not in entry_set:
        entry_set.add(entry)
        with open(logfile, 'a') as f:
            f.write(entry + '\n')
            f.flush()


def class_str_from_dict(class_name: str, d: Dict[str, Any]) -> str:
    return "{}({})".format(class_name, ", ".join(k+"="+str(v) for k,v in d.items()))


class ProcessKillingExecutor:
    """
    The ProcessKillingExecutor works like an `Executor <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with different arguments asynchronously.

    But other than the `ProcessPoolExecutor <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that terminates after the function returns or
    when a timeout occurs.

    This means that contrary to the Executors and similar classes provided by the Python Standard Library, you can
    rely on the fact that a process will get killed if a timeout occurs and that absolutely no side effects can occur
    between function calls.

    Note that descendant processes of each process will not be terminated – they will simply become orphaned.
    """

    def __init__(self, max_workers: int = None):
        """
        Initializes a new ProcessKillingExecutor instance.
        :param max_workers: The maximum number of processes that can be used to execute the given calls.
        """
        super().__init__()
        self.max_workers = (os.cpu_count() or 1) if max_workers is None else max_workers
        if self.max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self.manager = Manager()
        self.lock = Lock()
        self.cv = Condition()
        self.worker_count = 0

    def map(self, func: Callable, iterable: Iterable, timeout: float = None, callback_timeout: Callable = None,
            daemon: bool = True):
        """
        Returns an iterator (i.e. a generator) equivalent to map(fn, iter).
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task times out. It gets the same arguments as
                                 the original function
        :param daemon: run the child process as daemon
        :return: An iterator equivalent to: map(func, *iterables) but the calls may be evaluated out-of-order.
        """

        # approach:
        # create a fixed amount of threads, all threads share an input queue
        # the queue holds the function params and can be fairly short
        # each thread takes a function param from the queue, starts the process and joins
        # after the process has joined, the thread takes the next element from the queue
        # or terminates, if the shutdown flag was set (this is done by the main thread,
        # when the generator has no more elements)
        # the results are stored in an output queue, which the main thread flushes
        # and yields whenever a new place becomes free in the input queue
        # after the generator is empty, the main thread waits for all threads to terminate
        # and yields the remaining results.
        # issue: input order is not preserved with that approach. May not be that bad for me,
        # but this can be solved...

        # slight modification:
        # - we store the unfinished results in the output queue immediately. the threads do not alter the queues,
        #   they just mutate the result objects. therefore we don't even have to synchronize any queues
        # - whenever the main thread is notified, fill up the input queue, yield the next elements of the output queue,
        #   until it is empty or the next element is unfinished
        # that way we preserve order and don't interrupt processing.
        # Only issue: a long-running task might stall many short tasks, but that cannot be prevented when order has to be preserved...

        params = ({'func': func, 'args': args, 'timeout': timeout, 'callback_timeout': callback_timeout,
                   'daemon': daemon, 'result': Result()} for args in iterable)

        # supports indexing. not threadsafe. use append() and popleft()
        output_q = collections.deque()

        for thread_kwargs in params:
            # store result wrapper in output queue
            output_q.append(thread_kwargs['result'])
            # start the thread
            workers = self.__worker_count_inc()
            t = Thread(target=self.submit, kwargs=thread_kwargs)
            t.start()
            # yield all results from the output queue that are ready
            while len(output_q) > 0 and output_q[0].ready:
                yield output_q.popleft().value
            # blocks if max size is reached
            # there is the oh so slightest chance of a race condition here:
            # if the last thread calls notify just before we go into wait,
            # we have to wait for the next thread, which takes forever in a
            # single-thread scenario. Never happened so far, but still...
            if self.__worker_count_get() >= self.max_workers:
                with self.cv:
                    self.cv.wait()

        # almost done, wait for threads to finish, then yield the remaining results
        with self.cv:
            self.cv.wait_for(lambda: self.worker_count == 0)
        for result in output_q:
            yield result.value

    def submit(self, func: Callable = None, args: Any = (), kwargs: Dict = {}, result: 'Result' = None,
               timeout: float = None, callback_timeout: Callable[[Any], Any] = None,
               daemon: bool = True):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new process.
        Returns the result, if the process finished successfully, or None, if it fails or a timeout occurs.
        :param func: the function to execute
        :param args: the arguments to pass to the function. Can be one argument or a tuple of multiple args.
        :param kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function will be killed if it did not finish
        :param callback_timeout: this function will be called with the same arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or timed out
        """
        try:
            args = args if isinstance(args, tuple) else (args,)
            shared_dict = self.manager.dict()
            process_kwargs = {'func': func, 'args': args, 'kwargs': kwargs, 'share': shared_dict}
            p = Process(target=self._process_run, kwargs=process_kwargs, daemon=daemon)
            p.start()
            p.join(timeout=timeout)
            if 'return' in shared_dict:
                if result:
                    result.success(shared_dict['return'])
                return shared_dict['return']
            else:
                if result:
                    result.failure()
                if callback_timeout:
                    callback_timeout(*args, **kwargs)
                if p.is_alive():
                    p.terminate()
                return None
        except Exception as e:
            logger.error("Process failed due to exception: ", exc_info=1)
        finally:
            if result:
                result.ready = True
            self.__worker_count_dec()
            with self.cv:
                self.cv.notify()

    @staticmethod
    def _process_run(func: Callable[[Any], Any] = None, args: Any = (), kwargs: Dict = {}, share: Dict = None):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param args: the arguments to pass to the function
        :param kwargs: the kwargs to pass to the function
        :param share: a dictionary created using Manager.dict()
        """
        result = func(*args, **kwargs)
        share['return'] = result

    def __worker_count_inc(self):
        with self.lock:
            self.worker_count += 1
            return self.worker_count

    def __worker_count_dec(self):
        with self.lock:
            self.worker_count -= 1
            return self.worker_count

    def __worker_count_get(self):
        with self.lock:
            return self.worker_count


class Result:

    def __init__(self, value=None, ready=False):
        super().__init__()
        self.ready = ready
        self.value = value
        self.successful = None

    def success(self, value=None):
        self.value = value
        self.successful = True
        self.ready = True

    def failure(self, value=None):
        self.value = value
        self.successful = False
        self.ready = True


class ProgressLog:

    def __init__(self, logfile: str):
        super().__init__()
        self.logfile = logfile
        self.finished = self._parse(logfile)

    @classmethod
    def _parse(cls, logfile) -> Set[str]:
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf8') as fp:
                finished = []
                for line in fp.readlines():
                    if line:
                        split = line.split(' ', maxsplit=3)
                        if len(split) >= 2:
                            date, task_id = split[:2]
                            finished.append(task_id)
                return set(finished)
        return set()

    def add(self, task_id: str, message: str = ''):
        self.finished.add(task_id)
        self.__append_to_log(task_id, message)
        return self

    def __append_to_log(self, task_id: str, message: str = ''):
        with open(self.logfile, 'a', encoding='utf8') as fp:
            fp.write("{} {} {}\n".format(datetime.datetime.now().isoformat(), task_id, message))

    def __setitem__(self, name, value):
        self.__append_to_log(value)
        return self.finished.__setattr__(name, value)

    def __delitem__(self, name):
        return self.finished.__delattr__(name)

    def __contains__(self, value) -> bool:
        return self.finished.__contains__(value)

    def __str__(self):
        return "ProgressLog(logfile='{}', entries={})".format(self.logfile, self.finished)
