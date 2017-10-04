import util


def has_valid_chars(token: str) -> bool:
    """
    decides whether this token consists of a reasonable character mix.
    :param token: the token to inspect
    :return: True, iff the character mix is considered "reasonable"
    """
    hits = 0        # everything that is not alphanum or '-' or '.'
    limit = int(len(token) / 10)
    for c in token:
        if not (c.isalnum() or c == '.' or c == '-' or c == ' '):
            hits += 1
            if hits > limit:
                return False
    return True


def clean_parsed_text(text: str) -> str:
    # strip non-content lines, join adjacent lines, remove dashes, retain double line breaks

    # implementation: two-pass
    # 1. decide for each line individually whether it is a content line, write to bool array
    # 2. select and join adjacent content lines, write to new list
    # then join results

    lines = text.splitlines()
    include = [__is_content_line(line) for line in lines]
    clean = []
    first = True
    last_kept = False
    for line, keep in zip(lines, include):
        if keep:
            line_strip = line.strip()
            if last_kept:
                if clean[-1] == '-':
                    clean = clean[:-1]
                elif line_strip.startswith(('•','*')):
                    clean.append('\n')
                else:
                    clean.append(' ')
            elif first:
                first = False
            else:
                clean.extend('\n\n')
            clean.extend(line_strip)
            last_kept = True
        else:
            last_kept = False
    clean = ''.join(clean)
    return clean


def __is_content_line(line: str) -> bool:
    count = 0
    for char in line:
        if char.isalpha():
            count += 1
        if count > 2:
            return True
    return False


def strip_all_whitespace(text: str) -> str:
    return " ".join(text.split())


def fix_file_based_id(short_id: str) -> str:
    """
    repairs identifiers that have been extracted from file names.
    these lack the forward slash, which was part of the identifier in the early years of arxiv...
    :param short_id: the (possibly) broken identifier
    :return: the (hopefully) fixed identifier
    """
    if not short_id[0].isdigit():
        # oh boy, here we go
        if '/' not in short_id:
            # need to insert the slash before the first digit
            insert_at = 0
            for i, c in enumerate(short_id):
                if c.isdigit():
                    insert_at = i
                    break
            if insert_at > 0:
                return short_id[:insert_at] + '/' + short_id[insert_at:]
    # this is a regular identifier, or it has already been fixed
    return short_id


def __langdetect_demo():
    import langdetect, time
    entries = util.json_read_lines('../data/arxiv-meta-full-lines.json')
    t0 = time.time()
    for i, entry in enumerate(entries):
        title = entry['title'][0].replace("\n", " ")
        abstract = entry['description'][0].replace("\n", " ")
        langs = langdetect.detect_langs(title)  #  + ". " + abstract[:100]      # type: List[Language]
        if langs[0].lang != 'en' or langs[0].prob < 0.8:
            print("{}, {:.3f}, {}: \"{}\". Abstract: {}".format(langs[0].lang, langs[0].prob, entry['identifier'][0], title[:100], abstract[:100]))
        if i % 10000 == 0:
            t1 = time.time()
            print("10k done in {:.1f} s".format(t1-t0))
            t0 = t1
