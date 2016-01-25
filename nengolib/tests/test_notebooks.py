import fnmatch
import os
import warnings

import pytest

from nengo.utils.ipython import export_py, load_notebook
from nengo.utils.stdlib import execfile

_notebookdir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'doc', 'notebooks'))

_nbfiles = []
for root, dirs, fnames in os.walk(_notebookdir, topdown=True):
    dirs[:] = fnmatch.filter(dirs, '[!.]*')  # no hidden dirs
    for fname in fnmatch.filter(fnames, '[!_]*.ipynb'):  # no hidden notebooks
        _nbfiles.append(os.path.join(root, fname))


def _get_ipython():
    class MockNbRunner(object):
        def magic(self, s):
            warnings.warn("Skipping line: '%s'" % s)
    return MockNbRunner()


@pytest.mark.slow
@pytest.mark.parametrize('fname', _nbfiles)
def test_notebooks(fname, tmpdir, plt):
    import matplotlib
    matplotlib.use('Agg')
    import pylab as _pylab
    _pylab.show = lambda: warnings.warn("Skipping command: 'pylab.show()'")
    py = os.path.join(str(tmpdir), "notebook.py")
    export_py(load_notebook(fname), py)
    execfile(py, {'get_ipython': _get_ipython, 'pylab': _pylab})
