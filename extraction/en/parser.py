from nltk.tree import Tree


class DependencyNode(object):
    def __init__(self, idx, text, pos, dep, head):
        self.idx, self.text, self.pos, self.dep, self.head, self.childs = idx, text, pos, dep, head, []

    def __str__(self):
        return 'idx: {}, text: {}, pos: {}, parent: {}, relation: {}, childs: {}'.format(
            self.idx, self.text, self.pos, self.head, self.dep, ','.join(map(str, self.childs))
        )


class DependencyTree(object):
    def __init__(self):
        self.tree, self.root = [], None

    def __del__(self):
        del self.tree[:]
        self.tree, self.root = [], None

    def __str__(self):
        tokens, postags, deps, heads = [], [], [], []
        for node in self.tree:
            tokens.append(node.text)
            postags.append(node.pos)
            deps.append(node.dep)
            heads.append(node.head+1)
        s = 'text: {}\npos: {}\ndeps: {}\nheads: {}\n'
        return s.format(tokens, postags, deps, heads)

    def create(self, doc):
        del self.tree[:]
        tokens, postags = doc['words'], doc['pos']
        dependencies, heads = doc['predicted_dependencies'],  doc['predicted_heads']
        for i in range(len(tokens)):
            # head = -1 if heads[i] == 0 else heads[i]
            node = DependencyNode(idx=i, text=tokens[i], pos=postags[i], dep=dependencies[i], head=heads[i]-1)
            self.tree.append(node)
        for i, node in enumerate(self.tree):
            assert i == node.idx
            if node.head == -1:
                self.root = i
            else:
                self.tree[node.head].childs.append(node.idx)
        assert self.root is not None

    def subtree(self, index):
        queue, res = [index], []
        while queue:
            top = self.tree[queue[0]]
            queue.extend([child for child in top.childs if child != self.root])
            res.append(queue.pop(0))
        res.sort()
        return res


class DependencyAnalyser(object):
    def __init__(self, model, cuda=False):
        if cuda:
            model._model = model._model.cuda()
        self.parser_model = model

    def predict_batch(self, batch):
        instances = [self.parser_model._json_to_instance({"sentence": s}) for s in batch]
        outputs = self.parser_model.predict_batch_instance(instances)
        return outputs

    def predict_single(self, s):
        instance = self.parser_model._json_to_instance({"sentence": s})
        doc = self.parser_model.predict_instance(instance)
        return doc


class ConstituencyNode(object):
    def pformat(self, margin=70, indent=0, nodesep='', parens='()', quotes=False):
        s = self._pformat_flat(nodesep, parens, quotes)
        if len(s) + indent < margin:
            return s
        s = '%s%s%s' % (parens[0], self.label, nodesep)
        for child in self.childs:
            assert isinstance(child, ConstituencyNode)
            s += ('\n' + ' ' * (indent + 2) + child.pformat(margin, indent + 2, nodesep, parens, quotes))
        return s + parens[1]

    def _pformat_flat(self, nodesep, parens, quotes):
        childstrs = []
        for child in self.childs:
            assert isinstance(child, ConstituencyNode)
            if not child.is_leaf:
                childstrs.append(child._pformat_flat(nodesep, parens, quotes))
            else:
                childstrs.append(child.label)
        return '%s%s%s %s%s' % (parens[0], self.label, nodesep, ' '.join(childstrs), parens[1])

    def __init__(self, idx, label, is_leaf=False):
        self.idx, self.label, self.is_leaf = idx, label, is_leaf
        self.childs, self.parent = [], None

    def __str__(self):
        return self.pformat()

    def add_child(self, subtree):
        assert isinstance(subtree, ConstituencyNode)
        self.childs.append(subtree)
        subtree.parent = self

    def get_leaves(self, filters=None):
        leaves = []
        for child in self.childs:
            if filters is None or filters(child):
                if child.is_leaf:
                    leaves.append(child)
                else:
                    leaves.extend(child.get_leaves())
        return leaves

    def next_brother(self):
        try:
            assert self.parent is not None
        except AssertionError:
            print('root node has no parent')
        children = self.parent.children
        for i in range(len(children)):
            if children[i].id == self.idx:
                if i+1 >= len(children):
                    print('last child of parent has no next brother')
                    return None
                return children[i+1]


class ConstituencyTree(object):
    @staticmethod
    def equal(exist, generate):
        try:
            assert len(exist) == generate
            for i in range(len(exist)):
                assert exist[i] == generate
            return True
        except AssertionError:
            return False

    def __init__(self):
        self.tokens, self.postags, self.root, self.leaves, self.ids2leave = [], [], None, [], dict()

    def create(self, doc):
        del self.tokens[:]
        del self.postags[:]
        del self.leaves[:]
        del self.ids2leave
        self.tokens, self.postags, self.root, self.leaves = [], [], None, []
        tree, node_idx, q1, q2 = doc['trees'], 0, [], []
        _root = ConstituencyNode(idx=node_idx, label=tree.label())
        node_idx += 1
        q1.append(tree)
        q2.append(_root)
        while len(q1) != 0:
            length = len(q1)
            for i in range(length):
                r1 = q1.pop(0)
                r2 = q2.pop(0)
                if isinstance(r1, Tree):
                    for c in r1:
                        q1.append(c)
                        is_leaf = False if isinstance(c, Tree) else True
                        label = c.label() if isinstance(c, Tree) else c
                        n = ConstituencyNode(idx=node_idx, label=label, is_leaf=is_leaf)
                        node_idx += 1
                        r2.add_child(n)
                        q2.append(n)
        self.root, self.leaves = _root, _root.get_leaves()
        self.tokens, self.postags = [n.label for n in self.leaves], [n.parent.label for n in self.leaves]
        self.ids2leave = {self.leaves[i].idx: i for i in range(len(self.leaves))}
        # assert self.equal(self.tokens, doc['tokens']) and self.equal(self.postags, doc['pos_tags'])

    # def trace_root(self, idx):
    #     node = self.leaves[idx]
    #     node = node.parent.parent
    #     if node.label == 'VP':
    #         return node
    #     if node.label == 'NP':
    #         while node.parent is not None and node.parent.label == 'NP':
    #             node = node.parent
    #     assert node is not None
    #     return node
    def trace_root(self, idx):
        node = self.leaves[idx]
        while node is not None and node.label not in ['NP', 'VP']:
            node = node.parent
        assert node is not None
        return node


class ConstituencyAnalyser(object):
    def __init__(self, model, cuda=True):
        if cuda:
            model._model = model._model.cuda()
        self.parser_model = model

    def predict_batch(self, batch):
        instances = [self.parser_model._json_to_instance({"sentence": s}) for s in batch]
        outputs = self.parser_model._model.forward_on_instances(instances)
        return outputs

    def predict_single(self, s):
        instance = self.parser_model._json_to_instance({"sentence": s})
        doc = self.parser_model._model.forward_on_instance(instance)
        return doc


def visualize_instance(tree):
    try:
        assert type(tree) == Tree
        tree.draw()
    except AssertionError:
        print('assure input be instance of nltk.tree.Tree')

