# TODO: These perhaps belong elsewhere.
# Copied from misc/testutil.py.
# TODO: Copy or port over the tests.

class singleton(tuple):

    '''Create a single element tuple. Use to hold/wrap a value.

    >>> singleton('anything')
    singleton('anything')

    >>> x = {}; singleton(x)[0] is x;
    True
    '''

    __slots__ = ()

    def __new__(cls, value):
        return tuple.__new__(cls, (value,))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self[0])


class ReturnValue(singleton):

    '''Holder for value returned by a function call.

    >>> ReturnValue('anything')
    ReturnValue('anything')

    >>> x ={}; ReturnValue(x).value is x
    True

    >>> ReturnValue('anything').exception is None
    True
    '''

    __slots__ = ()
    exception = None

    @property
    def value(self):
        return self[0]


class ExceptionInstance(singleton):

    '''Holder for exception raised by function call.

    >>> ExceptionInstance('anything')
    ExceptionInstance('anything')

    >>> x = {}; ExceptionInstance(x).exception is x
    True

    >>> hasattr(ExceptionInstance('anything'), 'value')
    False

    '''
    __slots__ = ()

    # Deliberately not providing a value attribute.

    @property
    def exception(self):
        return self[0]

    def __eq__(self, other):

        if not isinstance(other, ExceptionInstance):
            return False
        return self.exception.args == other.exception.args

    def __ne__(self, other):

        return not self.__eq__(other)



# TODO: Add try_call.
def try_apply(fn, argv=[], kwargs={}):

    '''Execute fn, return either ReturnValue or ExceptionInstance.'''

    try:
        value = fn(*argv, **kwargs)
    except Exception as e:
        return ExceptionInstance(e)
    return ReturnValue(value)


def try_eval(*argv):
    '''Execute code, return either ReturnValue or ExceptionInstance.'''
    try:
        value = eval(*argv)
    except Exception as e:
        return ExceptionInstance(e)
    return ReturnValue(value)

