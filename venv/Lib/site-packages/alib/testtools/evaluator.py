'''Evaluate expressions
'''

from __future__ import absolute_import

import ast
import copy
import operator

from ..trytools import try_eval
from ..trytools import try_apply


__metaclass__ = type


# TODO: Warn of expressions being skipped.
def inspect(node):

    if isinstance(node, ast.Expr):

        value = node.value
        if type(value) is ast.Compare:
            return compare_factory(value)

        elif type(value) is ast.BinOp and type(value.op) is ast.Pow:
            return pow_factory(value)

        elif type(value) is ast.BoolOp:

            # TODO: Allow for more than 2 values.
            # NOTE: Test coverage and functionality must agree.
            if len(value.values) == 2:
                if isinstance(value.op, ast.Or):
                    return or_factory(value)
            return None


compare_dict = dict(
    # TODO: Finish this dict.
    Eq = operator.eq,
    In = lambda a, b: operator.contains(b, a),
    Is = operator.is_,
    Lt = operator.lt,
    IsNot = operator.is_not,
    NotEq = operator.ne,
    )


def get_args(*argv, **kwargs):
    return argv, kwargs


def eval_code_factory(l_dict, g_dict):

    # TODO: Don't modify a dictionary you don't own.
    l_dict['@get_args'] = get_args

    def eval_code(code):

        # TODO: Sort out l_dict, g_dict order mismatch.
        return try_eval(code, g_dict, l_dict)

    return eval_code


def compile_node(node):

    return compile(ast.Expression(node), '', 'eval')


def compile_nodes(nodes):

    return tuple(map(compile_node, nodes))


def compare_factory(node):

    lineno = node.lineno

    codes = compile_nodes([node.left] + node.comparators)
    ops = [type(op).__name__ for op in node.ops]

    def compare(eval_code):

        # TODO: Special case a single operation.
        # TODO: If you special case that, make sure you test.
        clean = True
        values = []
        for code in codes:
            val_or_exc = eval_code(code)
            if val_or_exc.exception:
                clean = False
            values.append(val_or_exc)

        comparisons = []
        for left, op, right in zip(values, ops, values[1:]):

            # TODO: Might raise exception.
            if left.exception or right.exception:
                comp = None
            else:
                comp = bool(compare_dict[op](left.value, right.value))
                if not comp:
                    clean = False
            comparisons.append(comp)

        if clean:
            return None
        else:
            return lineno, ops, values

    return compare


def pow_factory(node):

    lineno = node.lineno

    # TODO: I don't have much confidence in the logic of this code.

    left_c = compile_node(node.left)
    right_c = compile_node(node.right)

    # TODO: Parameter is callable that can evaluate code.
    def pow(eval_code):

        left, right = eval_code(left_c), eval_code(right_c)

        clean = True            # Redundant?

        if left.exception is None:
            if right.exception:
                return lineno, right.exception
            else:
                return lineno, 'Expected but did not get exception'

        if isinstance(left.exception, right.value):
            return None
        else:
            return lineno, left.exception, right.value

    return pow


def call_factory(root):

    lineno = root.lineno

    # TODO: Either remove copy or test that it works.
    root = copy.copy(root)      # We will change root.

    func_c = compile_node(root.func)

    # Getting the args of a function call is complex.
    root.func = ast.Name(
        id = '@get_args',
        ctx=ast.Load(),
        # Need to supply line_no and col_offset.
        lineno = 0,
        col_offset = 0
        )
    args_c = compile_node(root)

    # Create the closure function.
    def call(eval_code):

        func = eval_code(func_c)
        args = eval_code(args_c)

        if func.exception or args.exception:
            return lineno, 'Error setting up function call'
        else:
            return try_apply(func.value, *args.value)
    # Return the closure function.
    return call


def or_factory(root):

    lineno = root.lineno

    left_c, right_c = map(compile_node, root.values)

    def or_(eval_code):

        # Test order and code order correspond.
        left = eval_code(left_c)
        if left.exception:
            return lineno, 'Unexpected exception'
        else:
            if left.value:
                return None
            else:
                right = eval_code(right_c)
                if right.exception:
                    return lineno, 'Unexpected exception'
                else:
                    if right.value:
                        return None
                    else:
                        return lineno, 'false or false'

        return                  # Should not happen.

    return or_
