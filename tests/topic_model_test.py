from topic_model import Topic


def test_topic_serialization():
    root = Topic('root', layer=0)
    c1 = root.add_child('c1', [("foo", 0.5), ("bar", 0.3)])
    c1_1 = c1.add_child('c1-1', [("baz", 0.5)])
    c2 = root.add_child('c2', [("yolo", 0.5)])

    stored = root.store_recursively()
    topic_dict = Topic.restore_topics(stored)
    root2 = topic_dict['root']

    assert root.topic_id == root2.topic_id


def test_topic_model_stats():
    #topic_model.topic_stats('../data/arxiv-topics.json.bz2')
    pass
