#!/bin/bash

# Print the current directory
echo "Current directory: $(pwd)"

# Define main LaTeX file name (without extension)
MAIN_FILE="thesis"

# Define the folder where the .tex and .bib files are located
TEX_DIR="./tex"
BIB_DIR="./tex/files"

# Function to clean auxiliary files
clean_aux_files() {
    echo "Cleaning auxiliary files..."
    rm -f "$TEX_DIR/$MAIN_FILE".{aux,bbl,blg,log,toc,synctex.gz,out,fls,fdb_latexmk}
    rm -f "$BIB_DIR"/*.aux
    echo "Auxiliary files cleaned."
}

# Function to compile the LaTeX document
compile_latex() {
    echo "Compiling LaTeX document..."

    # Change directory to TEX_DIR to ensure LaTeX finds Tptesi2.cls
    cd "$TEX_DIR" || exit

    # Run pdflatex first pass
    pdflatex -synctex=1 -interaction=nonstopmode -file-line-error "$MAIN_FILE.tex"

    # Run BibTeX for bibliography
    bibtex "$MAIN_FILE.aux"

    # Run pdflatex twice more to ensure references and citations are correct
    pdflatex -synctex=1 -interaction=nonstopmode -file-line-error "$MAIN_FILE.tex"
    pdflatex -synctex=1 -interaction=nonstopmode -file-line-error "$MAIN_FILE.tex"

    echo "LaTeX document compiled."

    # Change back to the original directory
    cd - || exit
}

# Clean auxiliary files
clean_aux_files

# Compile the LaTeX document
compile_latex

# Final message
echo "PDF compilation finished. Check $TEX_DIR/$MAIN_FILE.pdf"
