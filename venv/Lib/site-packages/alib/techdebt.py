# TODO: This is complicated code necessary to test the full
# functionality.  Not needed for present application.  It is an
# example of how writing tests can be a useful force in software
# development.

# TODO: Refactor this to support inheritance.
# TODO: When we move this to alib.treetools, put it there and rewrite.

class Link_i_i:
    '''Increment statement line numbers by 3.
    '''
    def __init__(self):
        self.n = 0
        self.state = 0

    def inspect(self, node):
        assert self.state == 0
        self.state = 1
        self.curr = node
        return node.lineno

    def make_insertions(self):
        while 1:
            assert self.state == 1
            self.state = 0
            curr = self.curr
            curr.lineno += 3
            yield curr
