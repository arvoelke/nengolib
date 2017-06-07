# -*- coding: utf-8 -*-
#
# NengoLib documentation build configuration file, created by
# sphinx-quickstart on Mon May 22 13:35:46 2017.

import nengo
import nengolib

import inspect
import sys
from os.path import relpath, dirname

import matplotlib as mpl
import seaborn as sns  # noqa: F401
sns.set_style('white')

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'alabaster',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'nengo.utils.docutils',
]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = []
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'nengo': ('https://pythonhosted.org/nengo', None),
}

# -- numpydoc
numpydoc_use_plots = True
plot_include_source = True
plot_html_show_formats = False
plot_formats = [('png', 300)]
plot_rcparams = mpl.rcParams.copy()  # use the seaborn rcParams for examples
plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
old_show = plt.show
def new_show(tight=True, *args, **kwargs):
    if tight:
        plt.tight_layout()
    return old_show(*args, **kwargs)
plt.show = new_show
"""  # HACK: default to using tight_layout

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'nengolib'
copyright = u'2017, Aaron Voelker'
author = u'Aaron Voelker'
title = "{0} Documentation".format(project)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = nengolib.__version__
# The full version, including alpha/beta/rc tags.
release = nengolib.__version__

github_url = "https://github.com/arvoelke/nengolib/tree/v%s/nengolib" % (
    release)

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.

    Code borrowed from:
        https://github.com/numpy/numpy/blob/master/doc/source/conf.py
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(nengolib.__file__))

    return "%s/%s%s" % (
        github_url, fn, linespec)


# -- Options for HTML output ----------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    'logo': 'logo.png',
    'logo_name': True,
    'logo_text_align': 'center',
    'description': 'Tools for robust dynamics in Nengo.',
    'github_user': 'arvoelke',
    'github_repo': 'nengolib',
    'github_type': 'star',
    'travis_button': True,
    'codecov_button': True,
    'sidebar_width': '260px',
    'page_width': '1200px',
    'show_related': True,
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        'relations.html',
    ]
}

html_static_path = ['static']
