% ociamthesis v2.2 - Modified for Xiaohao Ji
%-------------------------- identification ---------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ociamthesis}[2010/11/22 v2.2 OCIAM thesis class]
%-------------------------- initial code -----------------------
\def\logoversion{squarelogo}
\DeclareOption{beltcrest}{\def\logoversion{beltcrest}}
\DeclareOption{shieldcrest}{\def\logoversion{shieldcrest}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[a4paper]{report}

\RequirePackage{graphicx} 
\RequirePackage{ifthen}   

\raggedbottom

% Define the default submitted text
\newcommand{\submittedtext}{{Dissertation zur Erlangung des Grades eines}}

% Declarations
\def\degreedate#1{\gdef\@degreedate{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\college#1{\gdef\@college{#1}}

% Setup Logo (Jiading - JD.png)
\def\logo{{\includegraphics[width=120mm]{JD.png}}}

% Text area and margins
\setlength{\topmargin}{0.0in}
\setlength{\oddsidemargin}{0.33in}
\setlength{\evensidemargin}{-0.08in}
\setlength{\textheight}{9.0in}
\setlength{\textwidth}{6.0in}

% --- Helper Environment ---
\newenvironment{alwayssingle}{%
       \@restonecolfalse
       \if@twocolumn\@restonecoltrue\onecolumn
       \else\if@openright\cleardoublepage\else\clearpage\fi
       \fi}%
       {\if@restonecol\twocolumn
       \else\newpage\thispagestyle{empty}\fi}

% REVISED TITLE PAGE (No Logo, All-Italic Degree Block, Better Spacing)
\renewcommand{\maketitle}{%
\begin{alwayssingle}
    \thispagestyle{empty}
  \null\vfill
  \begin{center}
    { \Huge {\bfseries {\@title}} \par}
    \vspace*{35mm} 
    {{\Large \bfseries \@author} \par}
    \vspace*{15mm}
    {\large {\@college} \par \vspace*{1ex} {Freie Universität Berlin} \par}
    \vspace*{35mm}
    % The degree block is now entirely italicized
    {\large \itshape {\submittedtext} \par \vspace*{1ex} {\@degree} \par} 
    \vspace*{30mm}
    {\large \@degreedate}
  \end{center}
  \null\vfill
\end{alwayssingle}}

% COMMAND FOR DEDICATED PICTURE PAGE
\newcommand{\makepicturepage}{%
\begin{alwayssingle}
    \thispagestyle{empty}
    \null\vfill
    \begin{center}
        \logo 
    \end{center}
    \vfill\null
\end{alwayssingle}}

% --- Environments (Preserved) ---

\newenvironment{dedication}
{\begin{alwayssingle} \thispagestyle{empty} \begin{center} \vspace*{1.5cm} {\LARGE } \end{center} \vspace{0.5cm} \begin{verse}\begin{center}}
{\end{center}\end{verse}\end{alwayssingle}}

\newenvironment{acknowledgements}
{\begin{alwayssingle} \thispagestyle{empty} \begin{center} \vspace*{1.5cm} {\Large \bfseries Acknowledgements} \end{center} \vspace{0.5cm} \begin{quote}}
{\end{quote}\end{alwayssingle}}

\newenvironment{originality}
{\begin{alwayssingle} \thispagestyle{empty} \begin{center} \vspace*{1.5cm} {\Large \bfseries Selbstständigkeitserklärung} \end{center} \vspace{0.5cm} \begin{quote}}
{\end{quote}\end{alwayssingle}}

\renewenvironment{abstract} 
{\begin{alwayssingle} \thispagestyle{empty} \begin{center} \vspace*{1.5cm} {\Large \bfseries Abstract} \end{center} \vspace{0.5cm} \begin{quote}}
{\end{quote}\end{alwayssingle}}

\newenvironment{abstractgermain} 
{\begin{alwayssingle} \thispagestyle{empty} \begin{center} \vspace*{1.5cm} {\Large \bfseries Zusammenfassung} \end{center} \vspace{0.5cm}}
{\end{alwayssingle}}

\newenvironment{romanpages}
{\cleardoublepage\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}