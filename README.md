# search-index-builder

This is a collection of scripts that should help you to build a topic model and a search index for the experimental search engine [epic search](https://github.com/Klamann/epic-search).

The main functions are:

* get metadata from [arxiv.org](https://arxiv.org/)
* pars pdfs and filter them by language
* create a topic model
* build an elasticsearch index

## Getting started

This application is just a collection of scripts, installation is not required. But to make sure that all dependencies are fulfilled, you should run

    python3 setup.py install --user

Also, you need a spacy model for the english language, if you want to build the index. To get the model:

    python3 -m spacy download en

If you're on Windows, installation of spacy, gensim, numpy, scipy, etc. might fail. Please use the whls from the *[Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/)* to get things up and running.

In case you're still missing any dependencies on Linux or Mac OSX, please install `libxml2` and `libxslt1`, e.g. on Debian:

    apt-get install libxml2-dev libxslt1-dev

## Usage

### arxiv-crawler

`arxiv-crawler.py` is a script that retrieves metadata about submitted papers over arxiv's [OAI](https://arxiv.org/help/oa/index) interface. It also also allows you to download small sets of papers directly from the public arxiv servers. Please note that by the terms of usage of arxiv.org you are *not* allowed to download large sets of documents from their webservice. You may retrieve a dump containing all data from [Amazon AWS](https://arxiv.org/help/bulk_data) (see [instructions below](#getting-the-arxiv-dataset)).

To get usage instructions:

    arxiv-crawler.py --help
    arxiv-crawler.py meta --help
    
To download metadata:

    arxiv_crawler.py meta --output FILE [--set SET] [--dateFrom DATE] [--dateUntil DATE]

the metadata will be written to the file specified after `--output` in json format. Use `.gz` or `.bz2` as file extension to enable compression.

You may choose to only download metadata for specific documents by using the `--set`, `--dateFrom` or `--dateUntil` arguments. See `arxiv-crawler.py meta --help` for detailed instructions on these.

If you want to download a few PDFs (e.g. for test purposes), you may use the `pdfs` command. But please don't use this tool to download more than a few files, it would cause significant load for the arxiv servers! If you want to get a large portion of the arxiv dataset, please read [the instructions for bulk data access](https://arxiv.org/help/bulk_data).

    arxiv_crawler.py pdfs -input FILE -output FOLDER

this command takes a metadata file (see `meta` command) and downloads all PDFs that are referenced in this file. The PDFs will be stored in the `--output` folder. *Use with caution!*

### PDF Parser

`pdf_parser.py` extracts the text contents from PDFs using the [pdfminer](http://www.unixuser.org/~euske/python/pdfminer/) library and stores the plain text in a JSON file.

Usage:

    pdf_parser.py -i FOLDER -o FOLDER [--tar] [-p N] [-t SECONDS] [-w N]

This command will parse all PDFs in the input folder and write the results to the specified output folder. The output folder will contain the plaintext representation of all PDFs in a single JSON file and several log files that contain progress information.

Because parsing of PDFs is rather slow (expect about 25 PDFs per minute per physical CPU core on a modern desktop CPU), this script tracks progress and can therefore be interrupted at any time and later continued without loss of data.

There are a few optional parameters for this task:

* `-p N`: limit the number of pages to parse per pdf (default: unlimited)
* `-t s`: parse timeout in seconds for a single pdf. This should be reasonably low, because PDFs are a mess and pdfminer won't be able to parse some of them in a reasonable amount of time, which will eventually block all your CPU cores (default: 60 seconds)
* `-w N`: the number of parallel worker threads to use for pdf parsing (default: 0, which means use all available cpu cores)

### Topic Model Generator

    usage: topic_model.py [-h] [-p FILE] [-t FILE] [-m FILE] -o PATH [-a]
                          [-T MODEL] [-l LAYERS] [-c LIMITS] [-M N] [-f LANG]
                          [-v N] [-d N]
    
    Build a nested topic model and classify documents
    
    optional arguments:
      -h, --help            show this help message and exit
      -p FILE, --input-pdfs FILE
                            path to the file containing the parsed PDFs (output of
                            pdf_parser)
      -t FILE, --input-tokens FILE
                            path to the file containing the tokens of the parsed
                            pdfs (optional, alternative to --input-pdfs)
      -m FILE, --input-meta FILE
                            path to the metadata file (output of arxiv_crawler.
                            required, if the category layer should be used)
      -o PATH, --output-prefix PATH
                            all output files, including temporary files, will be
                            prefixed with this string. all results will be stored
                            under this prefix aswell.
      -a, --abstracts-only  build topic models based on a paper's abstract only
                            (do not use the entire document text)
      -T MODEL, --topic-model MODEL
                            the topic model to use. Options: "hdp" (default),
                            "lda")
      -l LAYERS, --layers LAYERS
                            how many nested topic layers are to be used? Example:
                            "10,7,4"
      -c LIMITS, --limit-classification LIMITS
                            limits the number of topics that each document can be
                            assigned to at each layer during classification. One
                            number per layer, 0 stands for unlimited. Must have
                            same length as -l. Example: "1,2,0"
      -M N, --min-docs-per-topic N
                            require at least N documents per topic on each layer.
                            Can reduce the allowed topic count at each layer (but
                            never increase). Interrupts the build for a topic, if
                            less than 3*N documents remain (a topic model with
                            just two topics does not seem useful)
      -f LANG, --lang-filter LANG
                            filter by the specified language code. Defaults to
                            "en" (because we can currently only parse english
                            text)
      -v N, --vocab-size N  limit the size of the vocabulary, if specified
      -d N, --doc-limit N   just process the first N documents (useful for
                            testing)

### Elasticsearch Index Builder

`es_index.py` creates and populates an arxiv index from the crawled metadata, the generated topic model and the parsed full-text from the downloaded PDFs. The mapping is defined in `src/res/arxiv-mapping.json`. Please edit the index settings in this file, e.g. if you want to increase the default number of shards or replicas.

    usage: es_index.py [-h] [-H HOST] [-P PORT] [-u USER] [-p PASS] [-i NAME]
                       [-T NAME] [-n] [-o] [-m FILE] [-t FILE] [-c FILE]
    
    Create and populate an Elasticsearch index.
    
    optional arguments:
      -h, --help            show this help message and exit
      -H HOST, --host HOST  the network host of the Elasticsearch node (default:
                            localhost)
      -P PORT, --port PORT  the tcp port Elasticsearch is listening on (default:
                            9200)
      -u USER, --http-user USER
                            if http authentication is required, please specify the
                            name here
      -p PASS, --http-password PASS
                            if http authentication is required, please specify the
                            password here
      -i NAME, --index-name NAME
                            the name of the Elasticsearch index to use
      -T NAME, --index-type NAME
                            the type name of the Elasticsearch index to use
      -n, --new-index       create a new index (otherwise we assume an index
                            already exists)
      -o, --overwrite-index
                            overwrite an index of the same name, if it already
                            exists (otherwise we abort)
      -m FILE, --file-meta FILE
                            path to the metadata file (see `arxiv_crawler.py` to
                            get metadata). Can read bz2 and gzip compressed files.
      -t FILE, --file-topics FILE
                            path to the topic model file (see `topic_model.py` to
                            build a topic model). Can read bz2 and gzip compressed
                            files.
      -c FILE, --file-content FILE
                            path to the file containing the PDF texts (see
                            `pdf_parser.py` to extract text from PDFs). Can read
                            bz2 and gzip compressed files.

### Other Functions

`categories.py` converts the arxiv categories into the [1998 ACM Computing Classification System](http://www.acm.org/about/class/1998/). This function was used during the early stages of the prototype. It is only applicable to the computer science papers in the arxiv dataset and has now been superseded by the generic topic modeling tools provided by `topic_model.py`.

## Getting the arxiv dataset

The complete arxiv dataset is available from Amazon S3 in [requester pays buckets](https://docs.aws.amazon.com/AmazonS3/latest/dev/RequesterPaysBuckets.html). This means that you as a requester have to pay Amazon for the bandwidth, which is currently 0.09 US$ per GB. For the full 737 GB of PDFs (as of 2017-03), this results in 66 US$ plus VAT (in Germany this brings you to a total of 73€).

You can calculate the current price using the [AWS Calculator](https://calculator.s3.amazonaws.com/index.html) (select Amazon S3 on the left and enter the download size in the field *Data Transfer Out*)

Usually, I'd access the data using [aws-cli](https://github.com/aws/aws-cli), but there's an [issue](https://github.com/aws/aws-cli/issues/2557) with downloading data from requester pays buckets, which will hopefully be resolved soon. Anyway, here are the instructions for future reference:

    # install & configure
    pip install awscli
    aws configure
      AWS Access Key ID: foo
      AWS Secret Access Key: bar
      Default region name [us-west-2]: eu-central-1
      Default output format [None]: json
    aws configure set default.s3.max_concurrent_requests 20
    
    # list directories
    aws s3 ls s3://arxiv/pdf/ --request-payer requester
    
    # copy a single file or sync an entire directory (won't work until issue 2557 is resolved!)
    aws s3 cp --request-payer requester s3://arxiv/pdf/arXiv_pdf_1612_022.tar .
    aws s3 sync s3://arxiv/pdf/ . --dryrun --request-payer requester

If you want a working command line utility, use [s3cmd](http://s3tools.org/s3cmd). If you prefer a GUI application, [S3 Browser](http://s3browser.com/) may work for you (note: disable multipart downloads for vastly increased download speed).

## License

    Copyright 2016-2017 Sebastian Straub
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
