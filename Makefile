PKGNAME=graph-relax
SOURCES=$(wildcard *.py)

doc: README.md # CITATION.cff 
	pdoc -o docs graph_relax.py --docformat google


