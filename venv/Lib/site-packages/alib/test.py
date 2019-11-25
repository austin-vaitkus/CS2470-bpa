from __future__ import absolute_import

import os
import sys

from .testtools.testtools import testit
from .testtools.testtools import d_cond, f_cond, iter_d_f

if __name__ == '__main__':


    for name in sys.argv[1:]:
        if f_cond(os.path.basename(name)):
            testit(name)
        else:
            for d, f in iter_d_f(os.walk(name), d_cond, f_cond):
                testit(os.path.join(d, f))

