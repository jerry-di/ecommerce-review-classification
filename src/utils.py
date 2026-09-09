"""
utils - 通用工具
"""


def get_corpus(corpus_file):
    """获取指定文件中的语料（每行 文本\t标签）"""
    corpus = []
    with open(corpus_file, encoding='utf-8') as file_obj:
        content = file_obj.read()
    content = content.replace('\t\t', '\t')
    for line in content.splitlines():
        doc, label = line.split('\t', maxsplit=1)
        corpus.append((doc, int(label)))
    return corpus
