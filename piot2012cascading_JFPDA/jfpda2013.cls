%%%%%%% fichier adapt� de http://afia2007.imag.fr/instructions/

% Classe pf2007.cls adpat�e de pf2003 par Jean CHarlet
%
% Classe pf2003.cls adpat�e de ic2001 par Jean CHarlet
%
% Classe ic2001.cls adpat�e de EEGDRI3 par Jean CHarlet
%
% Classe EEGDRI3.cls adpat�e de ic2000 par Jean CHarlet
%
% Classe IC'2000 (ic2000.cls) par Jean Charlet
%
% adapt�e de la classe IC'99 (afia99.cls) d�velopp�e par Fabien Torre

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jfpda2013}[2013/04/01]
\LoadClass[10pt,twoside,a4paper]{article}

% Paquetages necessaires


% Le package vmargin ne marche pas bien...
%\RequirePackage{vmargin}
% ... je remplace par geometry
\usepackage[margin=30mm]{geometry}

\RequirePackage{amsthm}
\RequirePackage{times}
\RequirePackage{xspace}
\RequirePackage{graphicx}
% Pas toujours n�cessaire en fonction des param�tres de l'installation qui l'a
% parfois par d�faut mais �a marche. report� aussi sur le .tex
%\RequirePackage[latin1]{inputenc}

% Li�s aux installations. report� sur le fichier .tex
%\RequirePackage{french}
%\RequirePackage[french]{babel}
%\usepackage[T1]{fontenc}

%% Ouverture et fermeture du document
%% Impossible d'utiliser AtBeginDocument et AtEndDocument car french red�finit
%% lui m�me les begin et end de document... d'o� la ruse...
%% Ouverture et fermeture du document
%\let\BDocOri\document
%\def\document{\BDocOri
%\maketitle
%\renewcommand{\kwname}{{\bf Mots clef} :}
%}
%\let\EDocOri\enddocument
%\def\enddocument{\end{multicols}\EDocOri}



% Format papier et marges
% J'ai tout comment�, vu que j'utilise maintenant le package geometry.

%\setlength{\paperheight}{240mm}%
%\setlength{\paperwidth}{160mm}%
%\setlength{\textheight}{192mm}% plut�t que 192 sinon je ne vois pas o�
                              % on met les pages
%\setlength{\textwidth}{122mm}
%\setlength{\oddsidemargin}{15.6mm}
%\setlength{\evensidemargin}{20.6mm}
%\setlength{\topmargin}{15mm}% plut�t que 25 pour laisser de la place aux
                            % num�ros de page en bas


% Pour que les figure prennent la place qu'il faut sans rejeter le texte.
\renewcommand\topfraction{0.95}
\renewcommand\bottomfraction{0.95}
\renewcommand\textfraction{0.05}

\parindent=.3cm
% pour rendre les itemize un peu plus petits... �a marche ???
\renewcommand{\@listi}{\leftmargin\leftmargini
            \parsep 0\p@ \@plus\p@ \@minus\p@
            \topsep 6\p@ \@plus2\p@ \@minus4\p@
            \itemsep0\p@ \@plus\p@ \@minus\p@}

% g�rer les auteurs et leurs institutions
% piqu� ds llncs.cls de Springer-Verlag

% l�, je n'ai pas tout pris

\newcounter{@inst}
\newcounter{@auth}
\newcounter{auco}
\def\andname{and}
\def\lastandname{\unskip, and}
\newdimen\instindent

\def\institute#1{\gdef\@institute{#1}}

% Tir� de llncs nouvelle version qui fait dipara�tre les symboles ds les
% appels d'institution. Alors on fait pareil (lignes comment�es)

\newcommand\institutename{\par
  \begingroup
%  \parskip=\z@
  \parskip=1ex
  \parindent=\z@
  \setcounter{@inst}{1}%
  \def\and{\par\stepcounter{@inst}%
%    \noindent$^{\@fnsymbol\the@inst}$\enspace\ignorespaces}%
    \noindent$^{\the@inst}$\enspace\ignorespaces}%
  \setbox0=\vbox{\def\thanks##1{}\@institute}%
  \ifnum\c@@inst=1\relax
    \else
    \setcounter{footnote}{\c@@inst}%
    \setcounter{@inst}{1}%
%    \noindent$^{\@fnsymbol\the@inst}$\enspace
    \noindent$^{\the@inst}$\enspace
  \fi
  \ignorespaces
  \@institute\par
  \endgroup
}

\def\@fnsymbol#1{\ensuremath{\ifcase#1\or\star\or \dagger\or \ddagger
    \or{\star\star}\or {\star\star\star}
   \mathchar "278\or \mathchar "27B\or \|\or **\or \dagger\dagger
   \or \ddagger\ddagger \else\@ctrerr\fi}}

%\def\inst#1{\unskip$^{\@fnsymbol#1}$}
\def\inst#1{\unskip$^{#1}$}

%\def\subtitle#1{\gdef\@subtitle{#1}}
%\def\@subtitle{}

\renewcommand\maketitle{\par
  \begingroup
  \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
  \newpage
  \global\@topnum\z@   % Prevents figures from going at top of page.
  \@maketitle
  \thispagestyle{empty}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{\newpage
 \markboth{}{}%
 \def\lastand{\ifnum\value{@inst}=2\relax
                 \unskip{} \andname\
              \else
                 \unskip \lastandname\
              \fi}%
 \def\and{\stepcounter{@auth}\relax
          \ifnum\value{@auth}=\value{@inst}%
             \lastand
          \else
             \unskip,
          \fi}%
 \begin{center}%
   ~\vskip .8cm\
%   {\sffamily\scshape\Large \vskip .3cm\
%     \pretolerance=10000
%     TAGADA \par}\vskip .05cm
   {\LARGE \bfseries\boldmath
     \pretolerance=10000
     \@title \par}\vskip .3cm
   \setbox0=\vbox{\setcounter{@auth}{1}\def\and{\stepcounter{@auth}}%
     \def\thanks##1{}\@author}%
   \global\value{@inst}=\value{@auth}%
   \global\value{auco}=\value{@auth}%
   \setcounter{@auth}{1}%
   {\large\rm\lineskip .5em
     \@author\vskip.46cm}
   {\normalsize\rm\institutename}
 \end{center}%
 }

%% et pour faire comme llncs, le titre du r�sum� est ds le paragraphe avec une
%% d�finition de list...

\renewenvironment{abstract}{%
      \list{}{\advance\topsep by0.35cm\relax\small
      \leftmargin=1cm
      \labelwidth=\z@
      \listparindent=\z@
      \itemindent\listparindent
      \rightmargin\leftmargin}\item[\hskip\labelsep\bfseries\abstractname] :}
    {\endlist}

%  \newenvironment{abstract}{%
%      \titlepage
%      \null\vfil
%      \@beginparpenalty\@lowpenalty
%      \begin{center}%
%        \bfseries \abstractname
%        \@endparpenalty\@M
%      \end{center}}%
%     {\par\vfil\null\endtitlepage}

%% Les mots-cl�s : repiqu� avec un nom diff�rent sur french pour ne pas
%% entrer en conflit, dans le but de (1) corriger la faute d'orthographe de
%% certaines versions (pas de S) e t(2) de les envoyer � gauche au niveau dud
%% r�sum� 

\newcommand\motscles[1]{{\bfseries Mots-cl\'es} : #1}

% un titre court � donner pour les ent�tes impaires si celui d'origine est
% trop grand 40... et la m�me chose pour le titre de l'ouvrage sur les pages
% paires 

\def\shorttitle#1{\gdef\@shorttitle{#1}}
\def\@shorttitle{\@title}

\def\shortouvrage#1{\gdef\@shortouvrage{#1}}
\def\@shortouvrage{JFPDA~2013}

% le num�ro du chapitre en toute lettre
% Ne sert pas pour PUG

%\def\chapternumber#1{\gdef\@chapternumber{#1}}
%\def\chaptertot{\gdef\@chaptertot{toto}}
%\def\@chaptertot{{Chapitre \@chapternumber}}

% les ent�tes

\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@oddhead{\hfil\itshape\@shorttitle}%
    \def\@evenhead{\itshape\@shortouvrage\hfil}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }
\pagestyle{myheadings}

% Les titres en sans-serif qd n�cessaire

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {- 3.00ex\@plus -0.5ex \@minus -.15ex}%
                                    {1.2ex \@plus .1ex}%
                                    {\normalfont\normalsize\itshape}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\itshape}}


% pour le s�parateur des footnotes

\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width 5cm
  \kern2.6\p@}
\addtolength{\footnotesep}{2pt}

% Styles : page, biblio, theoreme

\bibliographystyle{jfpda2013}
\theoremstyle{break}

\renewenvironment{thebibliography}[1]
{\small
     {\section*{\refname
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
      \small\list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \itemsep0\p@\parsep3\p@ \@plus1\p@ \@minus\p@
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
}

% Allow short (name-less) citations, when used in
% conjunction with a bibliography style that creates labels like
%       \citename{<names>, }<year>
% Inutile si on fait du natbib

% \def\@cite#1#2{(#1\if@tempswa ,
% #2\fi)}
% \def\@biblabel#1{}

% \def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
%   \def\@citea{}\@cite{\@for\@citeb:=#2\do
%     {\@citea\def\@citea{; }\@ifundefined
%        {b@\@citeb}{{\bf ?}\@warning
%        {Citation `\@citeb' on page \thepage \space undefined}}%
% {\csname b@\@citeb\endcsname}}}{#1}}

% \let\@internalcite\cite
% \def\cite{\def\citename##1{##1}\@internalcite}
% \def\shortcite{\catcode`:=12\def\citename##1{}\@internalcite}
%\def\@biblabel#1{\def\citename##1{##1}[#1]\hfill} % from ``namen''
