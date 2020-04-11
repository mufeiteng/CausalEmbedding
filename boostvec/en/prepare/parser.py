# -*- coding: utf-8 -*-
class MyNode(object):
    def __init__(self, id, text, head, tag, dep):
        self.id, self.head, self.childs = id, head, []
        self.text, self.tag, self.dep = text, tag, dep

    def __str__(self):
        return 'id: {}, word: {}, postag: {}, parent: {}, relation: {}, children: {}'.format(
            self.id, self.text, self.tag, self.head, self.dep, ','.join(map(str, self.childs))
        )


class EnParseTree(object):
    def __init__(self):
        self.tree, self.root = [], None

    def create(self, doc):
        del self.tree[:]
        token_list = [token for token in doc]
        token_indices = {node: i for i, node in enumerate(token_list)}

        for i, token in enumerate(token_list):
            # parent = -1 if token.dep_ == 'ROOT' else token_indices[token.head]
            self.tree.append(MyNode(i, token.text, token_indices[token.head], token.tag_, token.dep_))
        for i, mynode in enumerate(self.tree):
            assert i == mynode.id
            if mynode.head == mynode.id:
                self.root = i
            if mynode.head != mynode.id:
                self.tree[mynode.head].childs.append(mynode.id)
        assert self.root is not None

    def subtree(self, index):
        queue, res = [index], []
        while queue:
            top = self.tree[queue[0]]
            queue.extend([child for child in top.childs if child != self.root])
            res.append(queue.pop(0))
        res.sort()
        return res
