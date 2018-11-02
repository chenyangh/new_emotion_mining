import bibtexparser
from titlecase import titlecase


def parse_name(names):

    name_list = names.split('and')
    ret_list = []
    for name in name_list:
        tmp = ''
        tokens = name.split(',')
        first_name = tokens[0]
        if len(tokens) == 1:
            continue
        rest = tokens[1]
        last_name = ''
        for part in rest.split('-'):
            last_name += list(part.strip())[0]
            last_name += '.'
        tmp += first_name
        if len(last_name) > 0:
            tmp += ', '
            tmp += last_name
        ret_list.append(tmp)
    return ','.join(ret_list)


def item_str(item):
    ret_val = '\\bibitem{' + item['ID'] + '} '
    ret_val += parse_name(item['author']) + ': '
    ret_val += titlecase(item['title'])
    if 'journal' in item:
        ret_val += ' ' + titlecase(item['journal'])
    if 'booktitle' in item:
        ret_val += ' ' + titlecase(item['booktitle'])
    if 'volume' in item:
        ret_val += ' ' + item['volume']
    if 'number' in item:
        ret_val += '(' + item['number'] + ')'
    if 'pages' in item:
        ret_val += ' ' + item['pages']
    if 'publisher' in item:
        ret_val += ' ' + titlecase(item['publisher'])
    if 'organization' in item:
        ret_val += ' ' + titlecase(item['organization'])
    if 'year' in item:
        ret_val += ' (' + item['year'] + ')'
    ret_val += '\n'
    return ret_val


if __name__ == '__main__':

    bibtex_str = open('data/cicling.bib', 'r').read()
    bib_database = bibtexparser.loads(bibtex_str)
    for item in bib_database.entries:
        print(item_str(item))
