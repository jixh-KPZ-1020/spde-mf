$pdf_mode   = 1;    # pdflatex -> PDF
$pdflatex   = 'pdflatex -shell-escape -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;    # run bibtex; don't fail build on bibtex warnings/errors
$out_dir    = 'build';
$silent     = 0;
