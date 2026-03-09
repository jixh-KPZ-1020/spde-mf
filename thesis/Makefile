# ===== Basic settings =====
MAIN      := main
BUILD_DIR := build
TEX       := $(MAIN).tex
PDF       := $(BUILD_DIR)/$(MAIN).pdf

LATEXMK    := latexmk
LATEX_OPTS := -pdf -f -outdir=$(BUILD_DIR)

# ===== Default target =====
.PHONY: all
all: pdf

# ===== Build PDF =====
.PHONY: pdf
pdf: $(PDF)

FIGURES := $(wildcard ../output/*.pdf)

$(PDF): $(TEX) $(wildcard Chapters/*.tex) $(wildcard Aux/*.tex) \
        biblio.bib ociamthesis.cls $(FIGURES)
	mkdir -p $(BUILD_DIR)/Aux $(BUILD_DIR)/Chapters
	$(LATEXMK) $(LATEX_OPTS) $(TEX) || test -f $@

# ===== Continuous preview build (opens PDF in VSCode tab) =====
.PHONY: watch
watch:
	mkdir -p $(BUILD_DIR)/Aux $(BUILD_DIR)/Chapters
	$(LATEXMK) $(LATEX_OPTS) -pvc -e '$$pdf_previewer = "code %S"' $(TEX)

# ===== Clean auxiliary files =====
.PHONY: clean
clean:
	$(LATEXMK) -outdir=$(BUILD_DIR) -c

# ===== Clean everything including PDF =====
.PHONY: distclean
distclean:
	$(LATEXMK) -outdir=$(BUILD_DIR) -C
	rm -rf $(BUILD_DIR)

# ===== Rebuild from scratch =====
.PHONY: rebuild
rebuild: distclean pdf

# ===== Open PDF (macOS) =====
.PHONY: open
open: pdf
	open $(PDF)

# ===== Archive submission package =====
.PHONY: archive
archive:
	tar -czf thesis-source.tar.gz \
	    main.tex conclusions.tex ociamthesis.cls biblio.bib \
	    Chapters Aux figures tree Makefile .latexmkrc
