# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
DOCSDIR       = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	\rm -rf "$(BUILDDIR)"
	\rm -rf _autosummary
	\rm -rf "$(DOCSDIR)"
	\rm -rf starsmashertools*.rst
	\rm -rf index.rst
	\rm -rf lib helpers mpl math

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(shell pwd)/build/html/index.html')"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@python3 create_indexrst.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	\cp -r "$(SOURCEDIR)/$(BUILDDIR)/html" "$(DOCSDIR)"
	\touch "$(DOCSDIR)/.nojekyll"
	@python3 fix_html.py
