import ast
from functools import partial
from itertools import repeat
import operator

# TODO: Migrate to alib.itertools (perhaps).
# TODO: Provide tests.
repeat_empty_tuple = repeat(())

def attrgettertuple(names):
    '''Always returns a tuple, whatever the length of argv.'''

    n = len(names)
    if n == 0:
        return repeat_empty_tuple
    getter = operator.attrgetter(*names)
    if n == 1:
        return lambda x: (getter(x),)
    else:
        return getter


# Not found an Python asdl parse I can use - need to build by hand
# what I need.  Doesn't take long.  Easier to do by hand than to write
# and test a program to do this.

BODY_TYPES = (
    # (name, attribute_names).

    ('Module', ('body',)),
    ('Interactive', ('body',)),
    ('Suite', ('body',)),
    ('FunctionDef', ('body',)),
    ('ClassDef', ('body',)),

    ('For', ('body', 'orelse')),
    ('While', ('body', 'orelse')),
    ('If', ('body', 'orelse')),

    ('With', ('body',)),

    ('Try', ('body', 'orelse', 'finalbody')), # Python 3 only.
    ('TryExcept', ('body', 'orelse')), # Python 2.6 and 2.7 only.
    ('TryFinally', ('body', 'finalbody')), # Python 2.6 and 2.7 only.
)

BODY_TYPE_LOOKUP = dict(
    (key, attrgettertuple(attribute_names))
    for key, attribute_names in BODY_TYPES
)


EXPR_TYPES = (
    # We're ommiting ctx.  Hope this is OK.
    ('Call', ('func', 'args', 'keywords', 'starargs', 'kwargs')),
    ('Name', ('id',)),
    ('NameConstant', ('value',)),
    ('Num', ('n',)),
)


EXPR_TYPE_LOOKUP = dict(
    (key, attrgettertuple(attribute_names))
    for key, attribute_names in EXPR_TYPES
)



def _generic_lookup(lookup, default, node):

    key = type(node).__name__
    fn = lookup.get(key, default)
    return fn(node)


# TODO: Add docstring.
'''Return tuple of the statement lists for the node.'''
get_statement_lists = partial(
    _generic_lookup,
    BODY_TYPE_LOOKUP,
    lambda node: ()
    )


get_expr_attributes = partial(
    _generic_lookup,
    EXPR_TYPE_LOOKUP,
    None                        # Raise exception.
    )


def parse_expr(s):
    '''Parse string and return corresponding expr.
    '''
    root = ast.parse(s, mode='eval')
    return root.body


# TODO: Only dependency is get_statement_lists - make it a parameter,
# say by using a function factory.  Once done, this belongs in
# alib.treetools.
def iter_statements(node):

    # Yield, in document order, the statements in the node.

    sls = get_statement_lists(node)

    for sl in sls:
        for i in range(len(sl)):
            yield sl, i         # So sl[i] is the current statement.

            # Recursion.  Unfortunately it is quadratic.
            for item in iter_statements(sl[i]):
                yield item


def replace(tree, inspect, insertions):

    insertions = iter(insertions)

    for sl, i in iter_statements(tree):

        # TODO: What about lineno and col_offset?
        removal = inspect(sl[i])
        if removal is None:
            continue
        yield removal
        sl[i] = next(insertions)
