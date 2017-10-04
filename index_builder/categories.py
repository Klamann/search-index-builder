"""
note: this code is now deprecated. use topic models instead.

read arxiv classes, convert to acm '98, build a proper data structure,
write more consistent classes to elasticsearch (drop existing),
keep class descriptions in a separate json (to be used by the webserver)

implementation notes:

- only include nodes with IDs
- (ignore nodes with <hasNote type="2"> as child) -> no, keep them, mark as deprecated
- each class consists of
  - id
  - label (optional)
  - parent id, child ids (optional)
  - related labels (optional, see <isRelatedTo>)
  - deprecated (boolean, see <hasNote type="2">, default false)
  - more specific classes (all nodes without ID below this one, in no specific order)

"""
import json
import re
from collections import OrderedDict
from typing import Dict, Union, Set, List

import xmltodict

import util

# arxiv:  https://arxiv.org/archive/cs
# acm:    http://www.acm.org/about/class/ccs98-html
corr_acm_mapping = {
    'Artificial Intelligence': ['I.2' ],
    'Computation and Language': ['I.2.7'],
    'Computational Complexity': ['F.1', 'F.2.3', 'F.4.3'],
    'Computational Engineering, Finance, and Science': ['J.2', 'J.3', 'J.4'],
    'Computational Geometry': ['I.3.5', 'F.2.2'],
    'Computer Science and Game Theory': ['F.4'],
    'Computer Vision and Pattern Recognition': ['I.2.10', 'I.4', 'I.5'],
    'Computers and Society': ['K.4'],
    'Cryptography and Security': ['D.4.6', 'E.3'],
    'Data Structures and Algorithms': ['F.2', 'E.1', 'E.2'],
    'Databases': ['H.2', 'E.2', 'J.1'],
    'Digital Libraries': ['H.3.7', 'I.7'],
    'Discrete Mathematics': ['G.2', 'G.3'],
    'Distributed, Parallel, and Cluster Computing': ['C.1.2', 'C.1.4', 'C.2.4', 'D.1.3', 'D.4.5', 'D.4.7', 'E.1'],
    'Emerging Technologies': [],
    'Formal Languages and Automata Theory': ['F.1.1', 'F.4.3'],
    'General Literature': [],
    'Graphics': ['I.3'],
    'Hardware Architecture': ['C.0', 'C.1', 'C.5'],
    'Human-Computer Interaction': ['H.1.2', 'H.5'],
    'Information Retrieval': ['H.3'],
    'Information Theory': ['E.4', 'H.1.1'],
    'Machine Learning': ['I.2.6'],
    'Learning': ['I.2.6'],  # alternative label for 'Machine Learning'
    'Logic in Computer Science': ['F.3'],
    'Mathematical Software': ['G.4'],
    'Multiagent Systems': ['I.2.11'],
    'Multimedia': ['H.5.1'],
    'Networking and Internet Architecture': ['C.2'],
    'Neural and Evolutionary Computing': ['I.2.6', 'I.5'],
    'Numerical Analysis': ['G.1'],
    'Operating Systems': ['D.4'],
    'Other': [],
    'Other Computer Science': [],
    'Performance': ['D.4.8', 'K.6.2'],
    'Programming Languages': ['D.1', 'D.3'],
    'Robotics': ['I.2.9'],
    'Social and Information Networks': ['H', 'J'],
    'Software Engineering': ['D.2'],
    'Sound': ['H.5.5'],
    'Symbolic Computation': ['I.1'],
    'Systems and Control': ['J.7'],
}

acm98_xml = "../data/categories/acmccs98-1.2.3.xml"
acm98_json = "../data/categories/acmccs98.json"


class Acm98:

    r_acm = re.compile(r'[A-K]\.([0-9m](\.[0-9m][0-9]?)?)?')

    def __init__(self, classes: Dict = None):
        super().__init__()
        self.classes = classes

    @classmethod
    def load(cls, json_file: str = None) -> 'Acm98':
        return cls._load_raw(json_file or acm98_json)._optimize()

    @staticmethod
    def _load_raw(json_file: str) -> 'Acm98':
        with(open(json_file, 'r')) as fp:
            acm_dict = json.load(fp)
            return Acm98(acm_dict)

    def _optimize(self) -> 'Acm98':
        self.classes = OrderedDict(sorted((k, v) for k, v in self.classes.items() if not v.get('deprecated')))
        for acm_class in self.classes.values():
            if 'label' in acm_class:
                # convert uppercase labels to title case, e.g. 'GENERAL' -> 'General'
                lbl = acm_class['label']
                if len(lbl) > 1 and lbl[1].isupper():
                    acm_class['label'] = lbl.title()
        return self

    def get_classes(self):
        return self.classes.values()

    def map_corr_labels(self, labels: List[str]) -> Set[str]:
        """
        maps CoRR labels to their closes ACM equivalents.
        if ACM labels are recognized, these are used instead of CoRR mappings.
        other labels will be ignored.
        about CoRR: https://arxiv.org/corr/subjectclasses
        :param corr: the CoRR class to convert
        :return: a set of the closest matching acm classes
        """
        # match acm
        acm_labels = set(x for x in labels if x and self.r_acm.fullmatch(x))
        if not acm_labels:
            # no acm labels -> convert corr
            for label in labels:
                corr_acm = self.map_corr_to_acm(label)
                acm_labels.update(corr_acm)
        return acm_labels

    @staticmethod
    def map_corr_to_acm(label: str) -> Set[str]:
        if label.startswith('Computer Science - '):
            corr = label.split(' - ', 1)
            acm = corr_acm_mapping.get(corr[1], [])
            return set(acm)
        else:
            return set()

    @classmethod
    def acm_xml_to_json(cls, xml_in, json_out):
        acm_tree = cls.parse_acm_98(xml_in)
        acm_flat = cls.flatten_acm_tree(acm_tree)
        #print(json.dumps(acm_flat, indent=2))
        with(open(json_out, 'w')) as fp:
            json.dump(acm_flat, fp, indent=2)

    @classmethod
    def parse_acm_98(cls, xml_file: str) -> Dict:
        with open(xml_file, 'r') as fp:
            doc = xmltodict.parse(fp.read())
            root = doc['node']
            acm_tree = cls.parse_node(root)
            acm_tree['id'] = '!root'
            return acm_tree

    @classmethod
    def parse_node(cls, xml_node: Dict) -> Dict[str, Union[Dict, str]]:
        node = {
            'id': xml_node['@id'],
            'label': xml_node.get('@label')
        }
        if 'isRelatedTo' in xml_node:
            related = xml_node['isRelatedTo'].get('node')
            node['related'] = related['@id'] if isinstance(related, dict) else [x['@id'] for x in related]
        if cls._is_deprecated(xml_node):
            node['deprecated'] = True
        if ('isComposedBy' in xml_node) and (not node.get('deprecated', False)):
            xml_children = xml_node['isComposedBy'].get('node', [])
            if isinstance(xml_children, dict):
                xml_children = [xml_children]
            unclassified_labels = util.truth_filter(child.get('@label') for child in xml_children
                                                    if '@id' not in child and not cls._is_deprecated(child))
            if unclassified_labels:
                node['unclassified_labels'] = unclassified_labels
            # recursive function call
            children = util.truth_filter(cls.parse_node(child) for child in xml_children if '@id' in child)
            if children:
                node['children'] = children
        return node

    @classmethod
    def flatten_acm_tree(cls, node: Dict, acc: Dict = None, parent: str = None) -> Dict:
        if not acc:
            acc = {}
        n = node.copy()
        acc[n['id']] = n
        if parent:
            n['parent'] = parent
        if 'children' in node:
            n['children'] = [x['id'] for x in node['children']]
            for child in node['children']:
                cls.flatten_acm_tree(child, acc, n['id'])
        return acc

    @staticmethod
    def _is_deprecated(node: Dict):
        return 'hasNote' in node and node['hasNote'].get('@type') == '2'


if __name__ == "__main__":
    #Acm98.acm_xml_to_json(acm98_xml, acm98_json)
    acm = Acm98.load(acm98_json)
    for c in acm.get_classes():
        print(c)
