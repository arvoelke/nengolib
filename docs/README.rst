Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the root directory of ``nengolib`` run::

    pip install -r docs/requirements.txt
    python setup.py build_sphinx

To render the Jupyter notebooks, you may also need to install ``Pandoc``,
e.g., by running::

    sudo apt-get install pandoc

Uploading the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone ``https://github.com/arvoelke/nengolib-docs``.

Replace the contents of ``nengolib-docs/*`` with the contents of ``nengolib/docs/build/html/*``. However, don't remove the files ``README.md`` or ``LICENSE`` from ``nengolib-docs``!

Within ``nengolib-docs``, commit and push the changes.
