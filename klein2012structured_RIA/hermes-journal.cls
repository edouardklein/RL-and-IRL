%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Document class file for articles published by Hermès Sciences
%%% Publications (http://www.hermes-science.fr/fr/).
%%%
%%% Author: Dominique Longin
%%% Toulouse University, CNRS, IRIT (UMR 5505)
%%% 118 route de Narbonne, 31062 Toulouse cedex 9, FRANCE
%%% Dominique.Longin@irit.fr
%%% Copyright 2009--2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Changes in class version 3.1.4b
%%% - ``Revue d'Intelligence Artificielle'' has been replaced by ``Revue d'intelligence artificielle'' in the footer of the first page
%%% Changes in class version 3.1.4
%%% - Before this version, captions were not well-formatted with subfig package use
%%% Changes in class version 3.1.3
%%% - Before this version, only one page was allowed for title, authors, ... DOI
%%% - the `page' string has been removed on the footer of the first page
%%% Changes in class version 3.1.2
%%% - figuretex environment added (traditional LaTeX figure environment)
%%% - jancl journal removed
%%% - string `Paris' removed from the DOI
%%% Changes in class version 3.1.1
%%% - problem in stared mathematical environments solved
%%% Changes in class version 3.1:
%%% - modification of space between end of the title and the \thanks mark
%%% - bug in \footnote fixed
%%% - string ``RSTI'' removed in both even headers and DOI for journals: RIA, DN, TSI, ISI
%%% - \sloppy command introduced both in \abstract and \resume commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IDENTIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% path of this file and of the bibliography style file
%%% (if this command is undefined path is the current directory)
\providecommand{\filespath}{./}

\newcommand{\className}{hermes-journal}
\newcommand{\classVersion}{3.1.4b}
\newcommand{\firstVersionDate}{2009/07/02}
\newcommand{\fileDate}{2012/06/07}
\newcommand{\docDate}{2012/06/04}


\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{\filespath\className}[\fileDate, version \classVersion, LaTeX2e document class for journal published by Hermes Sciences Publication]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIAL CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{ifthen}

%%% booleans declaration (by default, false)
\newboolean{BOOL@cropmarksEnabled}
\newboolean{BOOL@toc}
\newboolean{BOOL@foreword}
%%% \newboolean{BOOL@editorModeEnabled}
\newboolean{BOOL@englishEnabled}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DECLARATION OF OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareOption{cropmarks}{%
   \typeout{\className.cls: ``cropmarks'' option enabled}%
   \setboolean{BOOL@cropmarksEnabled}{true}%
}

\DeclareOption{toc}{%
   \typeout{\className.cls: ``toc'' option enabled}%
   \setboolean{BOOL@toc}{true}%
}

\DeclareOption{foreword}{%
   \typeout{\className.cls: ``foreword'' option enabled}%
   \setboolean{BOOL@foreword}{true}%
}

%%% \DeclareOption{editorMode}{%
%%%    \typeout{\className.cls: ``editorMode'' option enabled}%
%%%    \setboolean{BOOL@editorModeEnabled}{true}%
%%% }

\DeclareOption{english}{%
   \typeout{\className.cls: ``english'' option enabled}%
   \setboolean{BOOL@englishEnabled}{true}%
}

%%% processing of unknown options
\DeclareOption*{%
    \typeout{\className.cls WARNING: ``\CurrentOption'' option is unknown}
    \typeout{Valid options: cropmarks, editorMode, english}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXECUTION OF OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ProcessOptions\relax



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PACKAGE LOADING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\LoadClass[10pt,twoside,a4paper]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{amsthm}  % must be loaded AFTER amsmath package
\RequirePackage{mathrsfs}                       % gives \mathscr
\RequirePackage{epsfig}
\RequirePackage{pifont}
\RequirePackage{url}
\RequirePackage{color}
\RequirePackage{xspace}
\RequirePackage{calc}
\RequirePackage{times}
\ifthenelse{\boolean{BOOL@englishEnabled}}%
{%  THEN
    \RequirePackage[english]{babel}%
}{% ELSE
    \RequirePackage[frenchb]{babel}%
    \NoAutoSpaceBeforeFDP
    %\frenchbsetup{AutoSpacePunctuation=false}
}
\RequirePackage{apacite}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAIN CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%==============================================================
%==============================================================
% LAYOUT ======================================================
%==============================================================
%==============================================================


%==============================================================
% page layout =================================================
%==============================================================

% -------------------------------------------------------------
% horizontal margins
% -------------------------------------------------------------

\setlength{\voffset}{-1in}
\setlength{\oddsidemargin}{4.5cm}
\setlength{\evensidemargin}{4.5cm}
\setlength{\textwidth}{12cm plus 2pt minus 2pt}
\setlength{\marginparsep}{0cm}
\setlength{\marginparwidth}{0cm}
\setlength{\marginparpush}{0cm}

% -------------------------------------------------------------
% vertical margins
% -------------------------------------------------------------

\setlength{\hoffset}{-1in}
\setlength{\topmargin}{4.4cm}
\setlength{\headheight}{11pt}
\setlength{\headsep}{5.35cm-\topmargin-\headheight}
\setlength{\textheight}{19cm plus 3pt minus 2pt}
\setlength{\footskip}{5.35cm-4.4cm}


%==============================================================
% crop marks ==================================================
%==============================================================

% if cropmarks option is enabled
\ifthenelse{\boolean{BOOL@cropmarksEnabled}}%
{%  THEN
    \setlength{\overfullrule}{12pt}
    \newcommand{\DRAW@cropMarks}{%
        \begingroup
            \thinlines%
            \setlength{\unitlength}{1mm}%
            % \@drawCropMarks will called from the bottom of header line
            % Distance between this point and the top of the first text line is \headsep
            % But length cannot be used as argument of the initial picture position
            % \headsep = 5.35cm - 4.4cm - 11/72.27in = 0.95cm - 11*25.4/72.27cm = 0.5634cm = 5.634mm
            \begin{picture}(0,0)%
            \put(0,-5.634){\line(1,0){120}}      % top side of the text
            \put(0,-195.634){\line(1,0){120}}    % bottom side of the text
            \put(0,-5.634){\line(0,-1){190}}     % left side of the text
            \put(120,-5.634){\line(0,-1){190}}   % right side of the text
            \end{picture}
        \endgroup}%
        % the following command add a margin mark when line is too long (overfull \hbox)
        \setlength{\overfullrule}{1pt}
}{% ELSE
    \setlength{\overfullrule}{0pt}
    \newcommand{\DRAW@cropMarks}{}%
}


%==============================================================
% headers and footers =========================================
%==============================================================

% -------------------------------------------------------------
% header and footer of the first page of an article (which is ever an odd page)
% -------------------------------------------------------------

% normal issue
\newcommand{\ps@articleFirstPage}{%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\DRAW@cropMarks \WRITE@articleFreeColumn}}%
    \renewcommand{\@oddfoot}{\parbox{\textwidth}{\small\WRITE@journalName\
                                                 %. \WRITE@issueVolumeLabel\WRITE@issueVolume\
                                                 --
                                                 \WRITE@issueNumberLabel\WRITE@issueNumber/\WRITE@issueYear,
                                                 \WRITE@issuePagesInterval}}%
}

% toc
\newcommand{\ps@tocFirstPage}{%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\DRAW@cropMarks}}%
    \renewcommand{\@oddfoot}{}%
}

% foreword
\newcommand{\ps@forewordFirstPage}{%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\DRAW@cropMarks}}%
    \renewcommand{\@oddfoot}{\parbox{\textwidth}{}}%
}


% -------------------------------------------------------------
% header and footer of other pages
% -------------------------------------------------------------

% normal issue
\newcommand{\ps@articleOtherpages}{%
    \renewcommand{\@evenhead}{\parbox{\textwidth}{\DRAW@cropMarks{\small\thepage\ \ \ \ \ \WRITE@journalAcronym. \WRITE@issueVolumeLabel\WRITE@issueVolume\ -- \WRITE@issueNumberLabel\WRITE@issueNumber/\WRITE@issueYear}}}%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\DRAW@cropMarks\hspace*{\fill}{\small\WRITE@articleShortTitle\ \ \ \ \ \thepage}}}%
    \renewcommand{\@evenfoot}{}%
    \renewcommand{\@oddfoot}{}%
}

% special issue (there is a special issue as soon as an issue title is defined
% \VAR@issueTitle is an issue parameter. If it is defined, it must be printed in evenhead of the page
\newcommand{\ps@specialIssueArticleOtherpages}{%
%    \renewcommand{\@evenhead}{\parbox{\textwidth}{\DRAW@cropMarks{\small\thepage\ \ \ \ \ \WRITE@journalAcronym. \WRITE@issueVolumeLabel\WRITE@issueVolume\ -- \VAR@issueTitle}}}%
    \renewcommand{\@evenhead}{\parbox{\textwidth}{\DRAW@cropMarks{\small\thepage\ \ \ \ \ \WRITE@journalAcronym. \WRITE@issueVolumeLabel\WRITE@issueVolume\ -- \WRITE@issueNumberLabel\WRITE@issueNumber/\WRITE@issueYear. \VAR@issueTitle}}}%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\DRAW@cropMarks\hspace*{\fill}{\small\WRITE@articleShortTitle\ \ \ \ \ \thepage}}}%
    \renewcommand{\@evenfoot}{}%
    \renewcommand{\@oddfoot}{}%
}


%==============================================================
%==============================================================
% JOURNAL PARAMETERS ==========================================
%==============================================================
%==============================================================


%==============================================================
% journal name ================================================
%==============================================================

\newboolean{BOOL@journalNameDefined}

\newcommand{\journalName}[1]{
    \ifthenelse{\not\boolean{BOOL@journalNameDefined}}%
    {%  THEN
        \newcommand{\VAR@journalName}{#1}
        \setboolean{BOOL@journalNameDefined}{true}
    }{% ELSE
        % nothing else
    }
}

% -------------------------------------------------------------
% write the journal name
% (used in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@journalName}{%
    \ifthenelse{\boolean{BOOL@journalNameDefined}}%
    {%  THEN
        \VAR@journalName%
    }{% ELSE
        \textcolor{red}{!!!! Journal name undefined !!!!}
    }%
}

%==============================================================
% journal acronym =============================================
%==============================================================

\newboolean{BOOL@journalAcronymDefined}

\newcommand{\journalAcronym}[1]{
    \ifthenelse{\not\boolean{BOOL@journalAcronymDefined}}%
    {%  THEN
        \newcommand{\VAR@journalAcronym}{#1}
        \setboolean{BOOL@journalAcronymDefined}{true}
    }{% ELSE
        % nothing else
    }%
}

% -------------------------------------------------------------
% write the journal acronym
% (used in DOI and in even headers)
% -------------------------------------------------------------

\newcommand{\WRITE@journalAcronym}{%
    \ifthenelse{\boolean{BOOL@journalAcronymDefined}}%
    {%  THEN
        \VAR@journalAcronym%
    }{% ELSE
        \textcolor{red}{!!!! Journal acronym undefined !!!!}
    }%
}

%==============================================================
% English journal or French journal? ==========================
%==============================================================

%%% It is the same as the language of the journal name.

\newboolean{BOOL@journalHasEnglishTitleDefined}

% is the title of the journal in English? (by default: no)
\newboolean{BOOL@journalHasEnglishTitle}

\newcommand{\journalHasEnglishTitle}{
    \ifthenelse{\not\boolean{BOOL@journalHasEnglishTitleDefined}}%
    {%  THEN
        \setboolean{BOOL@journalHasEnglishTitle}{true}
        \setboolean{BOOL@journalHasEnglishTitleDefined}{true}
    }{% ELSE
        % nothing else
    }
}

\newcommand{\journalHasFrenchTitle}{
    \ifthenelse{\not\boolean{BOOL@journalHasEnglishTitleDefined}}%
    {%  THEN
        \setboolean{BOOL@journalHasEnglishTitleDefined}{true}
    }{% ELSE
        % nothing else
    }
}


%==============================================================
% journal DOI type ============================================
%==============================================================

%%% The normal form of DOI is built from both Volume and Pages:
%%% <volume>.<first page>-<last page>
%%% There is one volume per year. In some cases, the first page of
%%% an issue is always `1' for every issue in the volume. Thus, in this
%%% case, two different articles of the same volume can have the
%%% same DOI. To avoid this case, DOI has a special form:
%%% <volume>.<number>.<first page>-<last page>

\newboolean{BOOL@journalDoiTypeDefined}

% is DOI normal or special? (by default, normal)
\newboolean{BOOL@specialDOI}

\newcommand{\journalSpecialDoiType}{
    \ifthenelse{\not\boolean{BOOL@journalDoiTypeDefined}}%
    {%  THEN
        \setboolean{BOOL@specialDOI}{true}
        \setboolean{BOOL@journalDoiTypeDefined}{true}
    }{% ELSE
        % nothing else
    }
}

\newcommand{\journalNormalDoiType}{
    \ifthenelse{\not\boolean{BOOL@journalDoiTypeDefined}}%
    {%  THEN (DOI is normal by default)
        \setboolean{BOOL@journalDoiTypeDefined}{true}
    }{% ELSE
        % nothing else
    }
}

%==============================================================
% journal abstract and keywords language(s) ===================
%==============================================================

%%% Some journal have both abstract and keywords only in english.
%%% For other journals, abstract and keywords
%%% are both in French and in English: in this case, the first abstract
%%% is in French if the article is written in French, and the first
%%% abstract is in English is the article language is English.

\newboolean{BOOL@journalLanguagesNumberDefined}

% must abstract and keywords be written both in English and in French,
% or must they be written only in English? (by default, two)
\newboolean{BOOL@journalInEnglishOnly}

\newcommand{\journalInEnglishOnly}{
    \ifthenelse{\not\boolean{BOOL@journalLanguagesNumberDefined}}%
    {%  THEN
        \setboolean{BOOL@journalInEnglishOnly}{true}
        \setboolean{BOOL@journalLanguagesNumberDefined}{true}
    }{% ELSE
        % nothing else
    }
}

\newcommand{\journalNotOnlyInEnglish}{
    \ifthenelse{\not\boolean{BOOL@journalLanguagesNumberDefined}}%
    {%  THEN
        \setboolean{BOOL@journalLanguagesNumberDefined}{true}
    }{% ELSE
        % nothing else
    }
}

%==============================================================
%==============================================================
% ISSUE PARAMETERS ============================================
%==============================================================
%==============================================================

%%% Issue parameters must be initialized only one times. Thus, if
%%% those parameters are defined in the Editor File preamble, parameters
%%% of included article (authors' articles) will be ignored.

%==============================================================
% volume ======================================================
%==============================================================

\newboolean{BOOL@issueVolumeDefined}

\newcommand{\issueVolume}[1]{
    \ifthenelse{\not\boolean{BOOL@issueVolumeDefined}}%
    {%  THEN
        \newcommand{\VAR@issueVolume}{#1}
        \setboolean{BOOL@issueVolumeDefined}{true}
    }{% ELSE
        % nothing else
    }
}

% -------------------------------------------------------------
% write a label before the issue volume
% (used in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issueVolumeLabel}{%
    \ifthenelse{\boolean{BOOL@journalHasEnglishTitleDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@journalHasEnglishTitle}}%
        {%  THEN
            Volume\
        }{% ELSE
            Volume\
        }%
    }{% ELSE
        \textcolor{red}{!!!! Journal language undefined !!!!}
    }%
}

% -------------------------------------------------------------
% write the issue volume iff it has been defined
% (used in DOI and in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issueVolume}{%
    \ifthenelse{\boolean{BOOL@issueVolumeDefined}}%
    {%  THEN
        \VAR@issueVolume%
    }{% ELSE
        \textcolor{red}{!!!! Issue volume undefined !!!!}
    }%
}

%==============================================================
% number ======================================================
%==============================================================

\newboolean{BOOL@issueNumberDefined}

\newcommand{\issueNumber}[1]{
    \ifthenelse{\not\boolean{BOOL@issueNumberDefined}}%
    {%  THEN
        \newcommand{\VAR@issueNumber}{#1}
        \setboolean{BOOL@issueNumberDefined}{true}
    }{% ELSE
        % nothing else
    }
}

% -------------------------------------------------------------
% write a label before the issue number
% (used in even headers and in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issueNumberLabel}{%
    \ifthenelse{\boolean{BOOL@journalHasEnglishTitleDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@journalHasEnglishTitle}}%
        {%  THEN
            No.\
        }{% ELSE
            n\textsuperscript{o}\
        }%
    }{% ELSE
        \textcolor{red}{!!!! Journal language undefined !!!!}
    }%
}

% -------------------------------------------------------------
% write the issue number iff it has been defined
% (used in even headers, in special DOI, and in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issueNumber}{%
    \ifthenelse{\boolean{BOOL@issueNumberDefined}}%
    {%  THEN
        \VAR@issueNumber%
    }{% ELSE
        \textcolor{red}{!!!! Issue number undefined !!!!}
    }%
}

%==============================================================
% year ========================================================
%==============================================================

\newboolean{BOOL@issueYearDefined}

\newcommand{\issueYear}[1]{
    \ifthenelse{\not\boolean{BOOL@issueYearDefined}}%
    {%  THEN
        \newcommand{\VAR@issueYear}{#1}
        \setboolean{BOOL@issueYearDefined}{true}
    }{% ELSE
        % nothing else
    }
}

\newcommand{\WRITE@issueYear}{%
    \ifthenelse{\boolean{BOOL@issueYearDefined}}%
    {%  THEN
        \VAR@issueYear%
    }{% ELSE
        \textcolor{red}{!!!! Issue year undefined !!!!}
    }%
}


%%% %==============================================================
%%% % number of the first page ====================================
%%% %==============================================================
%%%
%%% \newboolean{BOOL@issueFirstPageNumberDefined}
%%%
%%% \newcommand{\issueFirstPage}[1]{
%%%     \ifthenelse{\not\boolean{BOOL@issueFirstPageNumberDefined}}%
%%%     {%  THEN
%%%         \newcommand{\VAR@issueFirstPageNumber}{#1}
%%%         \setboolean{BOOL@issueFirstPageNumberDefined}{true}
%%%     }{% ELSE
%%%         % nothing else
%%%     }
%%% }

%==============================================================
% title (in case of special issue) ============================
%==============================================================

\newboolean{BOOL@issueTitleDefined}

\newcommand{\issueTitle}[1]{
    \ifthenelse{\not\boolean{BOOL@issueTitleDefined}}%
    {%  THEN
        \newcommand{\VAR@issueTitle}{#1}
        \setboolean{BOOL@issueTitleDefined}{true}
    }{% ELSE
        % nothing else
    }
}

%==============================================================
% guest editors (in case of special issue) ====================
%==============================================================

\newboolean{BOOL@journalGuestEditorsDefined}

\newcommand{\editors}[1]{
    \ifthenelse{\not\boolean{BOOL@journalGuestEditorsDefined}}%
    {%  THEN
        \newcommand{\VAR@journalGuestEditors}{#1}
        \setboolean{BOOL@journalGuestEditorsDefined}{true}
    }{% ELSE
        % nothing else
    }
}


%==============================================================
%==============================================================
% JOURNAL AND ISSUE INITIALIZATION ============================
%==============================================================
%==============================================================

%==============================================================
% building of a journal-specific command of initialization
% IN: <journal initialization command> (generally, the journal acronym in small caps),
%     <journal acronym>, <journal name>, <journal title language>, <DOI type>, <abstract type>
% OUT: a command named <journal acronym> having 5 parameters
%      [special issue title], year, volume, issue, first page
%      and updating both journal acronym and journal name values
% EXAMPLE: \BUILT@journalCommand{rmpd}{RMPD}{Road Materials and Pavement Design}{journalHasEnglishTitle}{journalNormalDoiType}{journalInEnglishOnly}
%          produces a \newcommand{\rmpd}[5][NIL]{...} command
%==============================================================

\newcommand{\BUILT@journalCommand}[6]{%
    \expandafter\newcommand\csname#1\endcsname[4][NIL]{%
        \ifthenelse{\equal{##1}{NIL}}%
        {%  THEN
            % do nothing
        }{% ELSE
            \issueTitle{##1}
        }
        \issueYear{##2}
        \issueVolume{##3}
        \issueNumber{##4}
        \journalAcronym{#2}
        \journalName{#3}
        \csname #4\endcsname
        \csname #5\endcsname
        \csname #6\endcsname
    }
}

%==============================================================
% building of proper command for each journal
%==============================================================

% -------------------------------------------------------------
% Full English journals (no French abstract, no French keywords)
% -------------------------------------------------------------
%\BUILT@journalCommand{jancl}{JANCL}{Journal of Applied Non-Classical Logic}{journalHasEnglishTitle}{journalNormalDoiType}{journalInEnglishOnly}
      \BUILT@journalCommand{rmpd}{RMPD}{Road Materials and Pavement Design}{journalHasEnglishTitle}{journalNormalDoiType}{journalInEnglishOnly}

% -------------------------------------------------------------
% journals with English title but having both English and French articles
% -------------------------------------------------------------
        \BUILT@journalCommand{ejess}{EJESS}{European Journal of Economic and Social Systems}{journalHasEnglishTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
                                \BUILT@journalCommand{jds}{JDS}{Journal of Decision Systems}{journalHasEnglishTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
              \BUILT@journalCommand{ejcm}{EJCM}{European Journal of Computational Mechanics}{journalHasEnglishTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
\BUILT@journalCommand{ejece}{EJECE}{European Journal of Environmental and Civil Engineering}{journalHasEnglishTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
               \BUILT@journalCommand{ejee}{EJEE}{European Journal of Electrical Engineering}{journalHasEnglishTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}

% -------------------------------------------------------------
% journals with French title but having both English and French articles
% -------------------------------------------------------------
            \BUILT@journalCommand{imm}{I2M}{Instrumentation, Mesure, Métrologie}{journalHasFrenchTitle}{journalSpecialDoiType}{journalNotOnlyInEnglish}
             \BUILT@journalCommand{rig}{RIG}{Revue internationale de géomatique}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
    \BUILT@journalCommand{jesa}{JESA}{Journal européen des systèmes automatisés}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
\BUILT@journalCommand{rcma}{RCMA}{Revue des composites et des matériaux avancés}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
                    \BUILT@journalCommand{sdm}{SDM}{Santé, Décision, Management}{journalHasFrenchTitle}{journalSpecialDoiType}{journalNotOnlyInEnglish}
                               \BUILT@journalCommand{dn}{DN}{Document numérique}{journalHasFrenchTitle}{journalSpecialDoiType}{journalNotOnlyInEnglish}
          \BUILT@journalCommand{isi}{ISI}{Ingénierie des systèmes d'information}{journalHasFrenchTitle}{journalSpecialDoiType}{journalNotOnlyInEnglish}
              \BUILT@journalCommand{ria}{RIA}{Revue d'intelligence artificielle}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
                             \BUILT@journalCommand{ts}{TS}{Traitement du signal}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
             \BUILT@journalCommand{tsi}{TSI}{Technique et science informatiques}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}
                     \BUILT@journalCommand{hsp}{HSP}{Hermès Science Publication}{journalHasFrenchTitle}{journalNormalDoiType}{journalNotOnlyInEnglish}

%==============================================================
%==============================================================
% ARTICLE PARAMETERS ==========================================
%==============================================================
%==============================================================


%==============================================================
% article free column =========================================
%==============================================================

\newboolean{BOOL@articleFreeColumnDefined}

\newcommand{\freecolumn}[1]{
    \newcommand{\VAR@articleFreeColumn}{#1}
    \setboolean{BOOL@articleFreeColumnDefined}{true}
}

% -------------------------------------------------------------
% subtitle writing command used by \maketitle
% -------------------------------------------------------------

\newcommand{\WRITE@articleFreeColumn}{%
    \large\sc%
    \ifthenelse{\boolean{BOOL@articleFreeColumnDefined}}%
    {%  THEN
        \hspace*{\fill}\VAR@articleFreeColumn\hspace*{\fill}\par\smallskip\hrule
    }{% ELSE
        \hspace*{\fill}\strut\hspace*{\fill}\par\smallskip\hrule
    }
}


%==============================================================
% (short) article title =======================================
%==============================================================


\newcommand{\VAR@articleTitle}{!! Title undefined !!}
\newcommand{\VAR@articleShortTitle}{!! Short title undefined !!}

% -------------------------------------------------------------
% the author can specify the title of his article by the \title command
% [#1] is the short title (short title is used is the article header)
% #2 is the title
% -------------------------------------------------------------

\newboolean{BOOL@articleTitleDefined}

\renewcommand{\title}[2][NIL]{
    \renewcommand{\VAR@articleTitle}{#2}
    \ifthenelse{\equal{#1}{NIL}}%
    {%  THEN
        % a short title has not been specified
        % thus short title is the same as title
        \renewcommand{\VAR@articleShortTitle}{#2}
    }{% ELSE
        % a short title has been specified
        \renewcommand{\VAR@articleShortTitle}{#1}
    }
    \setboolean{BOOL@articleTitleDefined}{true}
}

% -------------------------------------------------------------
% title writing command used by \maketitle
% -------------------------------------------------------------

\newcommand{\WRITE@articleTitle}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0pt plus 1pt minus 1pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \fontsize{18}{20}\bf\selectfont
        \begin{flushleft}
        \ifthenelse{\boolean{BOOL@articleTitleDefined}}%
        {%  THEN
            \phantom{invisible line}
            \VAR@articleTitle%
        }{% ELSE
            \phantom{invisible line}
            \textcolor{red}{!!!! Article title undefined !!!!}
        }
        \end{flushleft}
        % new empty line
        \vspace{\baselineskip}%
    \endgroup
}

% -------------------------------------------------------------
% short title writing command used by header
% -------------------------------------------------------------

% Hermes requires that short title does not exceed 7.5cm

\newsavebox{\BOX@shortTitleBox}

\newcommand{\WRITE@articleShortTitle}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \small
        \sbox{\BOX@shortTitleBox}{\small\VAR@articleShortTitle\ \ \ \ \ \thepage}
        % IF the short title is too long
        \ifthenelse{\lengthtest{\wd\BOX@shortTitleBox > 80mm}}%
        {%  THEN
            \textcolor{red}{\ \ \ \ \ !!!! Short title too long !!!!}
        }{% ELSE
            \VAR@articleShortTitle
        }
    \endgroup
}


%==============================================================
% article subtitle =========================================
%==============================================================

\newboolean{BOOL@articleSubTitleDefined}

\newcommand{\subtitle}[1]{
    \newcommand{\VAR@articleSubTitle}{#1}
    \setboolean{BOOL@articleSubTitleDefined}{true}
}

% -------------------------------------------------------------
% subtitle writing command used by \maketitle
% -------------------------------------------------------------

\newcommand{\WRITE@articleSubTitle}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0pt plus 1pt minus 1pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \par\Large\bf              % \fontsize{14}{16}\selectfont
        \ifthenelse{\boolean{BOOL@articleSubTitleDefined}}%
        {%  THEN
            \begin{flushleft}
            \VAR@articleSubTitle
            % new empty line
            {\fontsize{18}{20}\bf\selectfont\vspace{\baselineskip}}
            \end{flushleft}
        }{% ELSE
            % nothing (subtitle is not obligatory)
        }
    \endgroup
}


%==============================================================
% article authors list ========================================
%==============================================================

% -------------------------------------------------------------
% contains the authors number. The value 0 means that no author
% has been defined
% -------------------------------------------------------------

\newcounter{COUNT@articleAuthorNumber}
\setcounter{COUNT@articleAuthorNumber}{0}

% -------------------------------------------------------------
% this command must be called as many as authors number.
% [#1]: number relying author and address(es)
% #2: both first name and last name of the author
% Output: commands named \VAR@articleAuthorI, \VAR@articleAuthorII...
% containing respectively the first author, the second author...
% -------------------------------------------------------------

\renewcommand{\author}[3][NIL]{%
    \stepcounter{COUNT@articleAuthorNumber}%
    \expandafter\newcommand\csname VAR@articleAuthor\Roman{COUNT@articleAuthorNumber}\endcsname{\mbox{#2~#3}}%
    \ifthenelse{\equal{#1}{NIL}}%
    {%  THEN
        \expandafter\newcommand\csname VAR@articleAuthorInstituteNumber\Roman{COUNT@articleAuthorNumber}\endcsname{}
    }{% ELSE
        \expandafter\newcommand\csname VAR@articleAuthorInstituteNumber\Roman{COUNT@articleAuthorNumber}\endcsname{\textsuperscript{{\normalfont \,#1}}}
    }
    % we create a false author containing the delimiter between this author and the next one
    % we do the same for the institute numbering
    \stepcounter{COUNT@articleAuthorNumber}
    \expandafter\newcommand\csname VAR@articleAuthor\Roman{COUNT@articleAuthorNumber}\endcsname{,\ }%
    \expandafter\newcommand\csname VAR@articleAuthorInstituteNumber\Roman{COUNT@articleAuthorNumber}\endcsname{}%
}

% -------------------------------------------------------------
% introduces a newline after the name of an author
% -------------------------------------------------------------

%%% in fact, each author name is followed by a 'false' \VAR@articleAuthor
%%%  command that contains ', ' (for linking this author and the next one).
%%%  The following command just replace ', ' by \newline in the current
%%% \VAR@articleAuthor command that is given by the current value of
%%% COUNT@articleAuthorNumber

\newcommand{\andauthor}{%
    \expandafter\renewcommand\csname VAR@articleAuthor\Roman{COUNT@articleAuthorNumber}\endcsname{, \newline}
}


% -------------------------------------------------------------
% author list writing command used by \maketitle
% -------------------------------------------------------------

\newcommand{\WRITE@articleAuthors}{%
    \begingroup
        \setlength{\parindent}{0pt}%
        \setlength{\parsep}{0pt plus 1pt minus 1pt}%
        \setlength{\baselineskip}{0pt plus 1pt minus 1pt}%
        \large\bf%              \fontsize{12}{14}\selectfont
        \raggedright%
        \ifthenelse{\value{COUNT@articleAuthorNumber}>0}%
        {%  THEN
            % the last author name command correspond to
            % a ', ' string. Thus, we erase this value by nothing
            \expandafter\renewcommand\csname VAR@articleAuthor\Roman{COUNT@articleAuthorNumber}\endcsname{}%
            %
            % we write the first author
            \par\VAR@articleAuthorI%
            \VAR@articleAuthorInstituteNumberI%
            %
            % we write the others
            \newcounter{COUNT@cptAuthors}%
            \setcounter{COUNT@cptAuthors}{1}%
            \whiledo{\value{COUNT@cptAuthors} < \value{COUNT@articleAuthorNumber}}%
            {%  DO
                \stepcounter{COUNT@cptAuthors}%
                \csname VAR@articleAuthor\Roman{COUNT@cptAuthors}\endcsname%
                \csname VAR@articleAuthorInstituteNumber\Roman{COUNT@cptAuthors}\endcsname%
            }%  END DO
        }{% ELSE
            \par\textcolor{red}{!!! No author defined !!!!}%
        }
        % new empty line
        \fontsize{18}{20}\bf\selectfont\vspace{\baselineskip}
    \endgroup
}

%==============================================================
% article biographies list ======================================
%==============================================================

% -------------------------------------------------------------
% contains the biography number. The value 0 means that no biography
% has been defined
% -------------------------------------------------------------

\newcounter{COUNT@articleBiographyNumber}
\setcounter{COUNT@articleBiographyNumber}{0}

% -------------------------------------------------------------
% this command must be called as many times as authors number.
% Contrary to authors, biography is automatically numbered.
% #1: biography of an author
% Output: commands named \VAR@articleBiographyI, \VAR@articleBiographyII, \VAR@articleBiographyIII...
% containing respectively the biography of the first author, of the second author...
% -------------------------------------------------------------

\newcommand{\biography}[1]{%
    \stepcounter{COUNT@articleBiographyNumber}
    \expandafter\newcommand\csname VAR@articleBiography\Roman{COUNT@articleBiographyNumber}\endcsname{#1}
}


% -------------------------------------------------------------
% addresses list writing command used by \AtEndDocument
% -------------------------------------------------------------

\newcommand{\WRITE@articleBiographies}{%
    \par\bigskip\begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \setlength{\parskip}{0pt plus 1pt minus 1pt}
        \normalsize\em    % \fontsize{10}{12}\selectfont
        \raggedright
        \newcounter{COUNT@cptBiographies}
        \setcounter{COUNT@cptBiographies}{0}
        \newcounter{COUNT@authorNum}
        \setcounter{COUNT@authorNum}{0}%
        \ifthenelse{\value{COUNT@articleBiographyNumber}>0}%
        {%  THEN
            \whiledo{\(\value{COUNT@cptBiographies}<\value{COUNT@articleBiographyNumber}\) \and
                     \(\value{COUNT@authorNum}<\value{COUNT@articleAuthorNumber}\)}%
            {%  DO
                \stepcounter{COUNT@cptBiographies}%
                \stepcounter{COUNT@authorNum}%
                {\normalfont\small\csname VAR@articleAuthor\Roman{COUNT@authorNum}\endcsname.}
                {\normalfont\small\itshape\csname VAR@articleBiography\Roman{COUNT@cptBiographies}\endcsname}\\[6pt]
                \stepcounter{COUNT@authorNum}% because after each author name there is a variable for author separator
            }%  END DO
            \ifthenelse{\(\value{COUNT@cptBiographies}=\value{COUNT@articleBiographyNumber}\) \and
                     \(\value{COUNT@authorNum}=\value{COUNT@articleAuthorNumber}\)}%
            {%  THEN
                % all is fine: there is the same number of author and of biographies
            }{% ELSE
                \newline\textcolor{red}{!!! The number of biographies is different of the number of authors !!!!}
            }
        }{% ELSE
%            \par\textcolor{red}{!!! No biography defined !!!!}
        }
    \endgroup
}


%==============================================================
% article addresses list ======================================
%==============================================================

% -------------------------------------------------------------
% contains the address number. The value 0 means that no address
% has been defined
% -------------------------------------------------------------

\newcounter{COUNT@articleAddressNumber}
\setcounter{COUNT@articleAddressNumber}{0}

% -------------------------------------------------------------
% this command must be called as many times as address number.
% Contrary to authors, address is automatically numbered.
% #1: address of an author(s group)
% Output: commands named \VAR@articleAddressI, \VAR@articleAuthorII, \VAR@articleAuthorIII...
% containing respectively the first author, the string ', ', the second author...
% -------------------------------------------------------------

\newcommand{\address}[2]{%
    \stepcounter{COUNT@articleAddressNumber}
    \expandafter\newcommand\csname VAR@articleAddress\Roman{COUNT@articleAddressNumber}\endcsname{#1}
    \expandafter\newcommand\csname VAR@articleEmail\Roman{COUNT@articleAddressNumber}\endcsname{\url{#2}}
}


% -------------------------------------------------------------
% addresses list writing command used by \maketitle
% -------------------------------------------------------------

\newcommand{\WRITE@articleAddresses}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \setlength{\parskip}{0pt plus 1pt minus 1pt}
        \normalsize\em    % \fontsize{10}{12}\selectfont
        \raggedright
        \ifthenelse{\value{COUNT@articleAddressNumber}>0}%
        {%  THEN
            \ifthenelse{\value{COUNT@articleAddressNumber}=1}%
            {%  THEN no number before address
                \par\csname VAR@articleAddressI\endcsname\\[3pt]
                \csname VAR@articleEmailI\endcsname
            }{% ELSE each address is numbered
                \newcounter{COUNT@cptAddresses}%
                \setcounter{COUNT@cptAddresses}{0}%
                \begin{list}%
                        {\arabic{COUNT@cptAddresses}.\hfill}%       SEE The \LaTeX\ Companion, pp.~61--62
                        {\setlength{\itemsep}{6pt}%                 vertical space between the different items
                         \setlength{\parsep}{0pt plus 1pt minus 1pt}
                         \setlength{\labelwidth}{10pt plus 1fil}%   width of the label box
                         \setlength{\labelsep}{0pt}%                space between label box and first line of item
                         \setlength{\leftmargin}{\labelwidth}%      space between left margin and the text of list
                        }
                \whiledo{\value{COUNT@cptAddresses} < \value{COUNT@articleAddressNumber}}%
                {%  DO
                    \stepcounter{COUNT@cptAddresses}%
                    \item
                    \csname VAR@articleAddress\Roman{COUNT@cptAddresses}\endcsname\\[3pt]
                    \csname VAR@articleEmail\Roman{COUNT@cptAddresses}\endcsname
                }%  END DO
                \end{list}
            }% end if
        }{% ELSE
            \par\textcolor{red}{!!! No address defined !!!!}
        }
        % new empty line (6pt)
        \vspace{\baselineskip}
    \endgroup
}


%==============================================================
% article abstract & résumé ===================================
%==============================================================

% -------------------------------------------------------------
% for every value of \boolean{BOOL@journalInEnglishOnly}
% English abstract is always defined
% -------------------------------------------------------------

\newboolean{BOOL@articleAbstractDefined}

\renewcommand{\abstract}[1]{%
    \newcommand{\VAR@articleAbstract}{#1}
    \setboolean{BOOL@articleAbstractDefined}{true}
}

% -------------------------------------------------------------
% French abstract (résumé) is only defined when
% \boolean{BOOL@journalInEnglishOnly} is false (normal case)
% -------------------------------------------------------------

\newboolean{BOOL@articleResumeDefined}

\ifthenelse{\not\boolean{BOOL@journalInEnglishOnly}}%
{%  THEN
    \newcommand{\resume}[1]{%
        \newcommand{\VAR@articleResume}{#1}
        \setboolean{BOOL@articleResumeDefined}{true}
    }
}{% ELSE
}

% -------------------------------------------------------------
% abstract (and résumé) writing command (used by \maketitle)
% -------------------------------------------------------------

%%% Algorithm:
%%% ----------
%%% IF abstract is english only
%%% THEN write english abstract only
%%% ELSE
%%%     IF English option is defined
%%%     THEN write English astract and then French résumé
%%%     ELSE write French résumé and then English abstract

\newcommand{\WRITE@journalAbstractResume}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{6pt plus 1pt minus 1pt}
        %   IF abstract is special
        \ifthenelse{\boolean{BOOL@journalInEnglishOnly}}%
        {%  THEN write english abstract only
            \ifthenelse{\boolean{BOOL@articleAbstractDefined}}%
            {%  THEN
                {\scriptsize\sc\em ABSTRACT.\ }%
                {\small\it\VAR@articleAbstract}\\[3pt]%
            }{% ELSE
                \textcolor{red}{{\small\it !!!! Abstract undefined !!!!}}%
            }%
        }{% ELSE abstract is not only in English but also in French
            \ifthenelse{\boolean{BOOL@articleAbstractDefined}\and\boolean{BOOL@articleResumeDefined}}%
            {%  THEN
                %   IF English option is defined
                \ifthenelse{\boolean{BOOL@englishEnabled}}%
                {%  THEN write English astract and then French résumé
                    {\scriptsize\sc\em ABSTRACT.\ }%
                    {\small\it\sloppy\VAR@articleAbstract}\\[3pt]%
                    {\scriptsize\sc\em R\'ESUM\'E.\ }%
                    {\small\it\sloppy\VAR@articleResume}\\[3pt]%
                }{% ELSE write French résumé and then English abstract
                    {\scriptsize\sc\em R\'ESUM\'E.\ }%
                    {\small\it\sloppy\VAR@articleResume}\\[3pt]%
                    {\scriptsize\sc\em ABSTRACT.\ }%
                    {\small\it\sloppy\VAR@articleAbstract}\\[3pt]%
                }%
            }{% ELSE
                \par\textcolor{red}{{\small\it !!!! Abstract AND Résumé must be defined but at least one or the other is missing !!!!}}%
            }%
        }%
    \endgroup
}

%==============================================================
% article keywords & mots-clés ================================
%==============================================================

% -------------------------------------------------------------
% for every value of \boolean{BOOL@journalInEnglishOnly}
% English keywords are always defined
% -------------------------------------------------------------

\newboolean{BOOL@articleKeywordsDefined}

\newcommand{\keywords}[1]{%
    \newcommand{\VAR@articleKeywords}{#1}
    \setboolean{BOOL@articleKeywordsDefined}{true}
}

% -------------------------------------------------------------
% French keywords (mots-clés) are only defined when
% \boolean{BOOL@journalInEnglishOnly} is false (normal case)
% -------------------------------------------------------------

\newboolean{BOOL@articleMotsClesDefined}

\ifthenelse{\not\boolean{BOOL@journalInEnglishOnly}}%
{%  THEN
    \newcommand{\motscles}[1]{%
        \newcommand{\VAR@articleMotsCles}{#1}
        \setboolean{BOOL@articleMotsClesDefined}{true}
    }
}{% ELSE
}

% -------------------------------------------------------------
% keywords (and mots-clés) writing command (used by \maketitle)
% -------------------------------------------------------------

%%% Algorithm:
%%% ----------
%%% IF keywords are special
%%% THEN write english keywords only
%%% ELSE
%%%     IF English option is defined
%%%     THEN write English keywords and then French mots-clés
%%%     ELSE write French mots-clés and then English keywords

\newcommand{\WRITE@journalKeywordsMotsCles}{%
    \begingroup
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        %   IF keywords are only written in English
        \ifthenelse{\boolean{BOOL@journalInEnglishOnly}}%
        {%  THEN write english keywords only
            \ifthenelse{\boolean{BOOL@articleKeywordsDefined}}%
            {%  THEN
                {\scriptsize\sc\em KEYWORDS:\ }%
                {\small\it\VAR@articleKeywords}\\[0pt]%
            }{% ELSE
                \textcolor{red}{{\small\it !!!! Keywords undefined !!!!}}%
            }%
        }{% ELSE
            \ifthenelse{\boolean{BOOL@articleKeywordsDefined}\and\boolean{BOOL@articleMotsClesDefined}}%
            {%  THEN
                %   IF English option is defined
                \ifthenelse{\boolean{BOOL@englishEnabled}}%
                {%  THEN write English astract and then French mots-clés
                    {\scriptsize\sc\em KEYWORDS:\ }%
                    {\small\it\VAR@articleKeywords}\\[3pt]%
                    {\scriptsize\sc\em MOTS-CL\'ES~:\ }%
                    {\small\it\VAR@articleMotsCles}\\[0pt]%
                }{% ELSE write French mots-clés and then English keywords
                    {\scriptsize\sc\em MOTS-CL\'ES~:\ }%
                    {\small\it\VAR@articleMotsCles}\\[3pt]%
                    {\scriptsize\sc\em KEYWORDS:\ }%
                    {\small\it\VAR@articleKeywords}\\[0pt]%
                }%
            }{% ELSE
                \par\textcolor{red}{{\small\it !!!! Keywords AND Mots-Clés must be defined but at least one or the other is missing !!!!}}%
            }%
        }%
    \endgroup
}


%==============================================================
% DOI =========================================================
%==============================================================

\newboolean{BOOL@articleDoiDefined}

\newcommand{\WRITE@articleDOI}{%
    \begingroup
        % the first page of the article (title, authors, addresses, abstract, keywords and DOI) cannot exceed one page
        % thus, just before writing DOI, current page number must be the first page number of the article
        \setlength{\parindent}{0pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \footnotesize\raggedleft%
        \ifthenelse{\boolean{BOOL@journalDoiTypeDefined}%
                    \or\boolean{BOOL@journalAcronymDefined}
                    \or\boolean{BOOL@issueVolumeDefined}%
                    \or\boolean{BOOL@issueNumberDefined}}%
        {% THEN
%            \par
            DOI:10.3166/\WRITE@journalAcronym.\WRITE@issueVolume.%
            % page number checking
            \label{@@@thispage}%
            % writing of the rest of DOI
            \ifthenelse{\boolean{BOOL@specialDOI}}%
            {%  THEN
                \WRITE@issueNumber.%
            }{% ELSE
                % nothing else
            }%
            \WRITE@issuePagesInterval\ \copyright\ \WRITE@issueYear\ Lavoisier
%%            % page checking (one page only)
%%            \ifthenelse{\value{COUNT@articleFirstPage}<\pageref{@@@thispage}}
%%            {%  THEN
%%                \newline
%%                \normalfont\normalsize\textcolor{red}{WARNING! Title, authors, addresses, ... and DOI cannot exceed one page!}
%%            }{% ELSE
%%            }
        }{% ELSE
            \par\textcolor{red}{{\small\it !!!! DOI number undefined (journal DOI type, journal acronym, issue volume, issue number, or article first page or last pasge is missing) !!!!}}%
        }%
%        \par\medskip
        \par
    \endgroup
}


%==============================================================
% article page numbers ========================================
%==============================================================


% -------------------------------------------------------------
% article first page number
% -------------------------------------------------------------

\newcounter{COUNT@articleFirstPage}

% default value = 1
\setcounter{COUNT@articleFirstPage}{1}

% this value can be updated by author
\newcommand{\firstpagenumber}[1]{%
    \setcounter{COUNT@articleFirstPage}{#1}
    \setcounter{page}{#1}
}

% -------------------------------------------------------------
% article last page number
% -------------------------------------------------------------

% article last page
\newcommand{\VAR@articleLastPage}{\pageref{LABEL@articleLastPage}}

\AtEndDocument{%
    \label{LABEL@articleLastPage}%
}

% -------------------------------------------------------------
% write a label before the pages interval
% (used in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issuePagesLabel}{%
    \ifthenelse{\boolean{BOOL@journalHasEnglishTitleDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@journalHasEnglishTitle}}%
        {%  THEN
            pages\
        }{% ELSE
            pages\
        }%
    }{% ELSE
        \textcolor{red}{!!!! Journal language undefined !!!!}
    }%
}

% -------------------------------------------------------------
% write the pages interval iff it has been defined
% (used in DOI and in first page footer)
% -------------------------------------------------------------

\newcommand{\WRITE@issuePagesInterval}{%
    \ifthenelse{\boolean{BOOL@issueNumberDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@journalHasEnglishTitleDefined}}%
        {%  THEN
            \ifthenelse{\boolean{BOOL@journalHasEnglishTitle}}%
            {%  THEN
                \arabic{COUNT@articleFirstPage}--\VAR@articleLastPage%
            }{% ELSE
                \arabic{COUNT@articleFirstPage}-\VAR@articleLastPage%
            }%
        }{% ELSE
            \textcolor{red}{!!!! Journal language undefined !!!!}
        }%
    }{% ELSE
        \textcolor{red}{!!!! Issue number undefined !!!!}
    }%
}


%==============================================================
% reception and acceptance dates =============================
%==============================================================

% -------------------------------------------------------------
% reception command definition
% -------------------------------------------------------------

\newboolean{BOOL@receptionDateDefined}

\newcommand{\receptionDate}[1]{
    \newcommand{\VAR@receptionDate}{#1}
    \setboolean{BOOL@receptionDateDefined}{true}
}

\newcommand{\WRITE@receptionDate}{%
    \ifthenelse{\boolean{BOOL@receptionDateDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            Received at \VAR@receptionDate\\
        }{% ELSE
            Reçu le \VAR@receptionDate\\
        }
    }{% ELSE
    }
}

% -------------------------------------------------------------
% acceptance command definition
% -------------------------------------------------------------

\newboolean{BOOL@acceptanceDateDefined}

\newcommand{\acceptanceDate}[2][NIL]{
    \ifthenelse{\equal{#1}{NIL}}%
    {%  THEN
        \newcommand{\VAR@acceptanceDateOptionnalField}{}
    }{% ELSE
        \newcommand{\VAR@acceptanceDateOptionnalField}{~#1}
    }
    \newcommand{\VAR@acceptanceDate}{#2}
    \setboolean{BOOL@acceptanceDateDefined}{true}
}

\newcommand{\WRITE@acceptanceDate}{%
    \ifthenelse{\boolean{BOOL@acceptanceDateDefined}}%
    {%  THEN
        \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            Accepted\VAR@acceptanceDateOptionnalField~at \VAR@acceptanceDate\\[12pt]
        }{% ELSE
            Accepté\VAR@acceptanceDateOptionnalField~le \VAR@acceptanceDate\\[12pt]
        }
    }{% ELSE
    }
}

% -------------------------------------------------------------
% reception and acceptation dates writing command used by \AtEndDocument
% -------------------------------------------------------------

\newcommand{\WRITE@receptionAcceptanceDates}{%
    \begingroup
        \bigskip
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0pt plus 1pt minus 1pt}
        \setlength{\parsep}{0pt plus 1pt minus 1pt}
        \par\normalsize\normalfont%
        \raggedleft
        \WRITE@receptionDate
        \WRITE@acceptanceDate
    \endgroup
}

%==============================================================
%==============================================================
% special environments ========================================
%==============================================================
%==============================================================


%==============================================================
% toc mode ====================================================
%==============================================================

\ifthenelse{\boolean{BOOL@toc}}%
{%  THEN
    % -------------------------------------------------
    % title
    % -------------------------------------------------
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            \title{Contents}%
        }{% ELSE
            \title{Table des matières}%
        }%

    \newcommand{\WRITE@beginToc}{%
        % -------------------------------------------------
        % headers and footers
        % -------------------------------------------------
        \ifthenelse{\boolean{BOOL@issueTitleDefined}}%
            {%  THEN
                \pagestyle{specialIssueArticleOtherpages}%
            }{% ELSE
                \pagestyle{articleOtherpages}%
            }
        %
        % -------------------------------------------------
        % \maketitle command
        % -------------------------------------------------
        \renewcommand{\maketitle}{}%
        %
        \vspace*{\fill}%
        \center
        %
        % journal name
        {\huge\VAR@journalName}
        %
        % content title and issue references (Volume - number/year)
        {\par\large\vspace*{4\baselineskip}%
         \VAR@articleTitle\hfill\WRITE@issueVolumeLabel\WRITE@issueVolume\ --\ \WRITE@issueNumberLabel\WRITE@issueNumber/\WRITE@issueYear%
         \par\vspace*{1\baselineskip}%
         %
         % if the issue is a special issue
         \ifthenelse{\boolean{BOOL@issueTitleDefined}}%
            {%  THEN
                \par\vspace*{2\baselineskip}%
                \normalfont\large\textsc{\VAR@issueTitle}%
                \par\vspace*{1\baselineskip}%
            }{% ELSE
                % nothing else
            }%
        }%
        %
        % header and footer for the first page
        \thispagestyle{tocFirstPage}\strut
        %
        \setlength{\parindent}{0mm}%
        \setlength{\parskip}{6pt}%
        %
        % -------------------------------------------------
        % toc columns width
        % -------------------------------------------------
        \newlength{\LEN@tocPagesMargin}
        \setlength{\LEN@tocPagesMargin}{0.8cm}
        \newlength{\LEN@tocSpaceMargin}
        \setlength{\LEN@tocSpaceMargin}{\LEN@tocPagesMargin}
        \newlength{\LEN@tocOtherMargin}
        \setlength{\LEN@tocOtherMargin}{\textwidth-\LEN@tocPagesMargin-\LEN@tocSpaceMargin-0.5\LEN@tocPagesMargin}
        %
        % -------------------------------------------------
        % toc input commands
        % -------------------------------------------------
        % counter of author for an article in the toc
        % (the aim is to manage the comma between authors)
        \newcounter{COUNT@authors}
        %
        % for the first author, no comma after
        % but for others, we add a comma BEFORE the each of them
        \renewcommand{\author}[3][]{%
            \stepcounter{COUNT@authors}%
            \ifthenelse{\value{COUNT@authors}=1}%
                {\normalfont\normalsize\textsc{##2~##3}}%
                {\unskip, \normalfont\normalsize\textsc{##2~##3}}%
        }
        %
        % we need sometimes to make a manual \\
        % In this case, no comma must be added before the next author
        % Thus, in this case, we consider him/her as the first author (of the new line)
        \renewcommand{\andauthor}{%
            \setcounter{COUNT@authors}{0}%
            \unskip,\\
        }
        %
        % The next command is use to write the toc
        % The aim is to copy and paste the authors commands of each article in the first field
        % (the optionnal parameter of the \author command included: it will be ignored)
        % {\author{author name}...} {title} {page}
        \newcommand{\paper}[3]{%
            \setcounter{COUNT@authors}{0}
            % page number
            \par\vspace{\baselineskip}\parbox[t]{\LEN@tocPagesMargin}{\raggedleft##3}%
            % space
            \parbox[t]{\LEN@tocSpaceMargin}{\strut}%
            % big box containing both title box and authors box
            \parbox[t]{\LEN@tocOtherMargin}{%
                \raggedright %
                % Title
                ##2\\[3pt]%
                % authors
                ##1%
            }
        }
    }%
%
    \newcommand{\WRITE@endToc}{%
        \vspace*{\fill}%
    }%
%
}{% ELSE toc
}


%==============================================================
% foreword mode ===============================================
%==============================================================

\ifthenelse{\boolean{BOOL@foreword}}%
{%  THEN

    % -------------------------------------------------
    % title
    % -------------------------------------------------
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            \title{Foreword}%
        }{% ELSE
            \title{Introduction}%
        }%


    \newcommand{\WRITE@beginForeword}{%
        % -------------------------------------------------
        % \maketitle command
        % -------------------------------------------------
        \renewcommand{\maketitle}{}%
        %
        % -------------------------------------------------
        % headers and footers (are the same as every article)
        % -------------------------------------------------
        \ifthenelse{\boolean{BOOL@issueTitleDefined}}%
            {%  THEN
                \pagestyle{specialIssueArticleOtherpages}%
            }{% ELSE
                \pagestyle{articleOtherpages}%
            }
        %
        % -------------------------------------------------
        % title writing
        % -------------------------------------------------
        \setlength{\parskip}{0.4\baselineskip plus 1pt minus 1pt}
        \par\vspace*{20pt}%
        \newsavebox{\BOX@forewordTitle}
        \sbox{\BOX@forewordTitle}{\normalfont\LARGE\textsc{\VAR@articleTitle}}
        \begin{flushright}
        \usebox{\BOX@forewordTitle}\\[-3pt]
        \rule{\wd\BOX@forewordTitle}{0.3pt}%
        \end{flushright}%
        \par\vspace*{\baselineskip}%
        %
        % header and footer for the first page
        \thispagestyle{forewordFirstPage}\strut
        %
        \setlength{\parindent}{5mm}%
        \setlength{\parskip}{6pt}%
        % fonts
        \normalfont\normalsize%
        %
    }%



    \newcommand{\WRITE@endForeword}{%
        % authors must be written
        % recall that command \author introduces a fake author between each names.
        % This fake author contains the delimiter between two successive authors.
        % Thus, we must skip it
        \begin{flushright}
        \begin{minipage}{8cm}
        \ifthenelse{\value{COUNT@articleAuthorNumber}>0 \and \value{COUNT@articleAddressNumber}>0}%
        {%  THEN
            % the last author name command correspond to
            % a ', ' string. Thus, we erase this value by nothing
            \expandafter\renewcommand\csname VAR@articleAuthor\Roman{COUNT@articleAuthorNumber}\endcsname{}%
            %
            % counter of the loop
            \newcounter{COUNT@cptAuthors}%
            \setcounter{COUNT@cptAuthors}{1}%
            \newcounter{COUNT@cptAddresses}%
            \setcounter{COUNT@cptAddresses}{0}%
            %
            % we open the array
            \whiledo{\(\value{COUNT@cptAuthors}<\value{COUNT@articleAuthorNumber}\) \and
                     \(\value{COUNT@cptAddresses}<\value{COUNT@articleAddressNumber}\)}%
            {%  DO
                \stepcounter{COUNT@cptAddresses}%
                %
                % we write the first author
                \raggedleft\normalfont\sc\csname VAR@articleAuthor\Roman{COUNT@cptAuthors}\endcsname\\[3pt]
                \raggedleft\normalfont\itshape\csname VAR@articleAddress\Roman{COUNT@cptAddresses}\endcsname\\
                \strut\\
                \stepcounter{COUNT@cptAuthors}%
                \stepcounter{COUNT@cptAuthors}%   two times to skip the false author
            }%  END DO
            %
            % Begin check addresses number vs authors number
            \ifthenelse{\(\value{COUNT@cptAuthors}<\value{COUNT@articleAuthorNumber}\) \and
                     \(\value{COUNT@cptAddresses}=\value{COUNT@articleAddressNumber}\)}%
            {%  THEN
                \textcolor{red}{!! More authors names than addresses have been defined !!}%
            }{% ELSE
                \ifthenelse{\(\value{COUNT@cptAuthors}>\value{COUNT@articleAuthorNumber}\) \and
                         \(\value{COUNT@cptAddresses}<\value{COUNT@articleAddressNumber}\)}%
                {%  THEN
                    \textcolor{red}{!! More addresses than authors names have been defined !!}%
                }{% ELSE
                }
            }
            % end check addresses number vs authors number
        }{% ELSE
            \par\textcolor{red}{!!! No author or no address  defined !!!!}
        }
        \end{minipage}
        \end{flushright}
    }% close \endForeword
%
}{% ELSE foreword
}


%==============================================================
%==============================================================
% FIRST PAGE ==================================================
%==============================================================
%==============================================================


%==============================================================
% \maketitle command ==========================================
%==============================================================

\renewcommand{\maketitle}{%
    % default header and footer selection
    % (MUST be placed after its definition and after definition of BOOL@issueTitleDefined)
    \ifthenelse{\boolean{BOOL@issueTitleDefined}}%
        {%  THEN
            \pagestyle{specialIssueArticleOtherpages}%
        }{% ELSE
            \pagestyle{articleOtherpages}%
        }
    \WRITE@articleTitle
    \thispagestyle{articleFirstPage}%
    \WRITE@articleSubTitle
    \WRITE@articleAuthors
    \WRITE@articleAddresses
    \hrule\strut\\[-4pt]
    \WRITE@journalAbstractResume
    \WRITE@journalKeywordsMotsCles
    \hrule\strut\\[-4pt]
    \WRITE@articleDOI
%    \newpage
    \normalfont\normalsize
    \setlength{\parindent}{5mm}
    \setlength{\parskip}{6pt}
}


%==============================================================
% \AtBeginDocument command ====================================
%==============================================================

\AtBeginDocument{%
    \ifthenelse{\boolean{BOOL@foreword}}%
    {%  THEN
        \WRITE@beginForeword%
    }{}
    \ifthenelse{\boolean{BOOL@toc}}%
    {%  THEN
        \WRITE@beginToc%
    }{}
    %% when subfig is used, captions must be well-formatted
    \@ifpackageloaded{subfig}{%
        % THEN
        \renewcommand{\figurename}{Figure}
        \captionsetup[figure]{justification=centering,singlelinecheck=false,labelsep=period,font={it}}
        \captionsetup[table]{justification=centering,singlelinecheck=false,labelsep=period,font={it}}
    }{
        % ELSE
    }
}


%==============================================================
%==============================================================
% LAST PAGE ==================================================
%==============================================================
%==============================================================

\AtEndDocument{%
    \ifthenelse{\boolean{BOOL@foreword} \or \boolean{BOOL@toc}}
    {%  THEN
        \ifthenelse{\boolean{BOOL@foreword}}
        {%  THEN
            \WRITE@endForeword
        }{}
        \ifthenelse{\boolean{BOOL@toc}}%
            {%  THEN
            \WRITE@endToc
        }{}
    }{% ELSE
        \WRITE@receptionAcceptanceDates

        \WRITE@articleBiographies
    }
}

%==============================================================
%==============================================================
% SECTIONS FORMATS ============================================
%==============================================================
%==============================================================


%%% Only four levels are avalaible
\setcounter{secnumdepth}{5}

%%% if a title is over several lines, there is no indentation
%%% after a line feed
\def\@hangfrom#1{%
       \setbox\@tempboxa\hbox{{#1}}%
       \hangindent \wd\@tempboxa\noindent\box\@tempboxa %% ORIGINAL de latex.ltx
       % \hangindent \wd 0 \noindent\box\@tempboxa %% old version of HERMES style
}

%%% Disable single lines at the start of a paragraph
\clubpenalty = 10000

%%% Disable single lines at the end of a paragraph
\widowpenalty = 10000
\displaywidowpenalty = 10000


%==============================================================
% Sections numbering ==========================================
%==============================================================

\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
\renewcommand{\theparagraph}{\thesubsubsection.\arabic{paragraph}}
\renewcommand{\thesubparagraph}{}

%==============================================================
% section command redefinition ================================
%==============================================================

%%% beforeskip = 18pt, afterskip = 12pt
\newlength{\LEN@beforeSection}
\newlength{\LEN@afterSection}

\renewcommand{\section}{%
    %%% we must take away the parskip value to the skips values
    \setlength{\LEN@beforeSection}{-\parskip}
    \addtolength{\LEN@beforeSection}{18pt}
    \setlength{\LEN@afterSection}{-\parskip}
    \addtolength{\LEN@afterSection}{12pt}
    \renewcommand{\@seccntformat}[1]{\thesection.\hspace{0.5em}}%
    \@startsection{section}%        nom de l'inter
        {1}%                        niveau de l'inter
        {0pt}%                      l'indentation du titre et du texte suivant
        {\the\LEN@beforeSection plus 2pt minus 2pt}%  beforeskip
        {\the\LEN@afterSection plus 1pt minus 1 pt}%  afterskip
        {\normalfont\bfseries\raggedright}%    style
}


%==============================================================
% section reference command ===================================
%==============================================================

\ifthenelse{\boolean{BOOL@englishEnabled}}%
    {%  THEN
        \newcommand{\sectionref}[1]{Section~\ref{#1}}
    }{% ELSE
        \newcommand{\sectionref}[1]{section~\ref{#1}}
    }%


%==============================================================
% subsection command redefinition =============================
%==============================================================

%%% beforeskip = 18pt, afterskip = 12pt
\newlength{\LEN@beforeSubSection}
\newlength{\LEN@afterSubSection}

\renewcommand{\subsection}{%
    %%% we must take away the parskip value to the skips values
    \setlength{\LEN@beforeSubSection}{-\parskip}
    \addtolength{\LEN@beforeSubSection}{18pt}
    \setlength{\LEN@afterSubSection}{-\parskip}
    \addtolength{\LEN@afterSubSection}{12pt}
    \renewcommand{\@seccntformat}[1]{{\normalfont\bfseries\itshape\thesubsection.\hspace{0.5em}}}%
    \@startsection%
        {subsection}%                       nom de l'inter
        {2}%                                niveau de l'inter
        {0pt}%                              l'indentation du titre et du texte suivant
        {\the\LEN@beforeSubSection plus 2 pt minus 2pt}%        beforeskip
        {\the\LEN@afterSubSection plus 1pt minus 1pt}%          afterskip (0pt  remove the line feed)
        {\normalfont\bfseries\itshape\raggedright}%    style
}

%==============================================================
% subsubsection command redefinition ==========================
%==============================================================

%%% beforeskip = 6pt, afterskip = 6pt
\newlength{\LEN@beforeSubSubSection}
\newlength{\LEN@afterSubSubSection}

\renewcommand{\subsubsection}{%
    %%% we must take away the parskip value to the skips values
    \setlength{\LEN@beforeSubSubSection}{-\parskip+6pt}
    \addtolength{\LEN@beforeSubSubSection}{6.01pt}
    \setlength{\LEN@afterSubSubSection}{-\parskip}
    \addtolength{\LEN@afterSubSubSection}{6.01pt}
    \renewcommand{\@seccntformat}[1]{{\normalsize\normalfont\itshape\thesubsubsection.\hspace{0.5em}}}%
    \@startsection%
        {subsubsection}%            nom de l'inter
        {3}%                        niveau de l'inter
        {0mm}%                      l'indentation du titre et du texte suivant
        {\the\LEN@beforeSubSubSection plus 1 pt minus 1pt}%        beforeskip
        {\the\LEN@afterSubSubSection plus 1pt minus 1pt}%          afterskip (0pt  remove the line feed)
        {\normalfont\normalsize\itshape\raggedright}%     style
}

%==============================================================
% paragraph command redefinition ==============================
%==============================================================

%%% beforeskip = 6pt, afterskip = 6pt
\newlength{\LEN@beforeParagraph}
\newlength{\LEN@afterParagraph}

\renewcommand{\paragraph}{%
    %%% we must take away the parskip value to the skips values
    \setlength{\LEN@beforeParagraph}{-\parskip+6pt}
    \addtolength{\LEN@beforeParagraph}{6.01pt}
    \setlength{\LEN@afterParagraph}{-\parskip}
    \addtolength{\LEN@afterParagraph}{6.01pt}
    \renewcommand{\@seccntformat}[1]{{\normalsize\normalfont\theparagraph.\hspace{0.5em}}}%
    \@startsection%
        {paragraph}%                nom de l'inter
        {4}%                        niveau de l'inter
        {0mm}%                      l'indentation du titre et du texte suivant
        {\the\LEN@beforeParagraph plus 1pt minus 1 pt}%  beforeskip
        {\the\LEN@afterParagraph plus 1pt minus 1pt}%    afterskip  (0pt remove the line feed)
        {\normalfont\raggedright}%             style
}

%==============================================================
% subparagraph command redefinition ===========================
%==============================================================

%%% beforeskip=\parskip, afterskip=1em horizontal
\renewcommand{\subparagraph}{%
    \renewcommand{\@seccntformat}[1]{}%
        \@startsection%
        {subparagraph}%             nom de l'inter
        {5}%                        niveau de l'inter
        {0cm}%                      l'indentation du titre et du texte suivant
        {0pt plus 1pt minus 1 pt}%  beforeskip
        {-1em}%                     afterskip  (0pt remove the line feed)
        {\normalfont\scshape\raggedright}%     style
}


%==============================================================
% appendix commands redefinition ===========================
%==============================================================

\renewcommand\appendix{\par
    \setcounter{section}{0}%
    \setcounter{subsection}{0}%
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            \gdef\thesection{Annex~\Alph{section}}%
        }{% ELSE
            \renewcommand{\thesection}{Annexe~\Alph{section}}
        }%
        \renewcommand{\thesubsection}{\Alph{section}.\arabic{subsection}}
}

\newcommand{\annex}{\appendix}

%==============================================================
% appendix commands reference =================================
%==============================================================

\ifthenelse{\boolean{BOOL@englishEnabled}}%
    {%  THEN
        \newcommand{\annexref}[1]{\ref{#1}}
    }{% ELSE
        \newcommand{\annexref}[1]{\ref{#1}}
    }%


%==============================================================
%==============================================================
% LIST ENVIRONMENTS ===========================================
%==============================================================
%==============================================================


%==============================================================
% itemize =====================================================
%==============================================================

% -------------------------------------------------------------
% itemize level counter
% -------------------------------------------------------------
% there is only 2 different levels. Thus the counter is either 1 or 2.
\newcounter{COUNT@currentListLevel}
\setcounter{COUNT@currentListLevel}{0}

% -------------------------------------------------------------
% itemize margins computation
% -------------------------------------------------------------
% See LaTeX companion 2nd Edition, p.\ 145
% After the end of the first line, the second line begins
%   at the left margin level. To do that, we construct a box
%   of the right width.
% STEP A
% For the first level of itemize,
%   the symbol is --
%   the distance between left margin and the symbol is 5mm
% For the second level of itemize,
%   the symbol is -
%   the distance between left margin and the symbol is 10mm
% => we add the distance between left margin and the width of the symbol
% STEP B : Finally, we add to the width of the box
%   the distance between the itemize symbol and the text
%
% #1: it is -- or - (depending on the level of the itemize)
\newcommand{\SET@itemizeIndent}[1]{%
    % horizontal spacing
    % ------------------
    \setlength{\leftmargin}{0pt}    % space between list and left margin
    \setlength{\rightmargin}{0pt}   % space between list and right margin
    \setlength{\listparindent}{0pt} % additional space before first lines of new paragraphs in a same item
    \setlength{\labelsep}{0.5em}    % space from first word of item to right side of label
    % STEP A : size of the label box:
    \settowidth{\labelwidth}{#1}%
    %  IF we are in first level of list
    \ifthenelse{\value{COUNT@currentListLevel} < 2}%
        {%  THEN
            \addtolength{\labelwidth}{5mm}%
        }{% ELSE
            \addtolength{\labelwidth}{10mm}%
        }
    % STEP B : if we add the 5mm (resp. 10mm) here, label will be raggedleft
    \setlength{\itemindent}{\labelwidth+\labelsep} %
    %
    % vertical spacing
    % ----------------
    \setlength{\topsep}{0pt}        % space between text and top of list
    \setlength{\partopsep}{3pt}     % additional space before list when list begins a new paragraph
    \setlength{\parsep}{3pt}        % space between paragraphs in the same item
    \setlength{\itemsep}{0pt}       % space between items
    %
    % penalitities
    \sloppy%
    \widowpenalty4000%              % single line at the page beginning
    \clubpenalty4000%               % single line at the page end
}

% -------------------------------------------------------------
% itemize environment
% -------------------------------------------------------------
% it is redefined at begin document by babel package. Thus:
% - STEP A:  we define a new environment
% - STEP B: we rename it as `itemize' at the begin of document
%
% STEP A:
\newenvironment{ENV@itemize}%
    {%  BEGIN PART
        \setlength{\parskip}{3pt}
        % IF we are entering in a list that have an imbrication level greater than 1
        \ifthenelse{\value{COUNT@currentListLevel} > 1}
            {%  THEN we generate an error
                \typeout{}
                \typeout{!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!}
                \typeout{!! ERROR: lists can only have two levels !!}
                \typeout{!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!}
                \@toodeep
            }{% ELSE
                % we increase the list level
                \stepcounter{COUNT@currentListLevel}
                \begin{list}%
                    {%  ARG 1 : the label used is \labelitemi or \labelitemii (depending on current list level)
                        \csname labelitem\roman{COUNT@currentListLevel}\endcsname
                    }{% ARG 2 : we set the margins and labelitemi(i)
                        % labels setting
                        \renewcommand{\labelitemi}{--}
                        \renewcommand{\labelitemii}{-}
                        %
                        % margins setting
                        \ifthenelse{\value{COUNT@currentListLevel} = 1}%
                            {%  THEN
                                \SET@itemizeIndent{\labelitemi}
                            }{% ELSE
                                \SET@itemizeIndent{\labelitemii}
                            }%
                    }% end of the arguments list
            }% of else
    }{% END PART
        \end{list}%
        \addtocounter{COUNT@currentListLevel}{-1}%
    }
%
% STEP B:
\AtBeginDocument{%
    \renewenvironment{itemize}{\begin{ENV@itemize}}{\end{ENV@itemize}}
}


%==============================================================
% enumerate ===================================================
%==============================================================

% enumerate uses the same level variable as itemize environment

\newenvironment{ENV@enumerate}%
    {%  BEGIN PART
        \setlength{\parskip}{3pt}
        % IF we are entering in a list that have an imbrication level greater than 1
        \ifthenelse{\value{COUNT@currentListLevel} > 1}
            {%  THEN we generate an error
                \typeout{}
                \typeout{!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!}
                \typeout{!! ERROR: lists can only have two levels !!}
                \typeout{!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!}
                \@toodeep
            }{% ELSE
                % we increase the list level
                \stepcounter{COUNT@currentListLevel}
                \begin{list}%
                    {%  ARG 1 : the label used is \labelenumi or \labelenumii (depending on current list level)
                        \csname labelenum\roman{COUNT@currentListLevel}\endcsname
                    }{% ARG 2 : we set the margins and labelitemi(i)
                        % we define which counter must be used
                        \usecounter{enum\roman{COUNT@currentListLevel}}
                        %
                        % labels setting
%                        \renewcommand{\theenumi}{\arabic{enumi}}%      default value
%                        \renewcommand{\labelenumi}{\theenumi.}%        default value
%                        \renewcommand{\theenumii}{\alph{enumii}}%      default value
                        \renewcommand{\labelenumii}{\theenumii)}
%                        \renewcommand{\p@enumii}{\theenumi.}% \ref{label} in a second level enumeration write \p@enumii\theenumii
                        %
                        % margins setting
                        \ifthenelse{\value{COUNT@currentListLevel} = 1}%
                            {%  THEN
                                \SET@itemizeIndent{\labelenumi}%    spaces are the same as those of itemize
                            }{% ELSE
                                \SET@itemizeIndent{\labelenumii}
                            }%
                    }% end of the arguments list
            }% of else
    }{% END PART
        \end{list}%
        \addtocounter{COUNT@currentListLevel}{-1}%
    }
%
% STEP B:
\AtBeginDocument{%
    \renewenvironment{enumerate}{\begin{ENV@enumerate}}{\end{ENV@enumerate}}
}



%==============================================================
%==============================================================
% INSERT COMMANDS =============================================
%==============================================================
%==============================================================


%==============================================================
% formulas ====================================================
%==============================================================

\setlength{\jot}{0.5\baselineskip}      % space between different lines of eqnarray
\setlength{\abovedisplayskip}{0mm}      % additional space before big formulas
\setlength{\abovedisplayshortskip}{0mm} % the same for short formulas
\setlength{\belowdisplayskip}{0mm}      % additional space after big formulas
\setlength{\belowdisplayshortskip}{0mm} % the same for short formulas


%==============================================================
% footnotes ===================================================
%==============================================================

\providecommand*{\dotFFN}{.}
\providecommand*{\kernFFN}{\kern 0.5em}
\footnotesep=8pt

\long\def\@makefntextFB#1{%
    \rule\z@\footnotesep
    \noindent\@thefnmark\dotFFN\kernFFN #1
}%

\setlength{\skip\footins}{7mm plus 2mm}

\renewcommand{\footnoterule}{\vspace*{-3pt}%
    \hrule width 2.5cm height 0.4pt \vspace*{3pt}}

\AtBeginDocument{\@ifpackageloaded{bigfoot}{}%
                  {% ELSE
                   \long\def\@makefntext#1{\@makefntextFB{#1}}%
                  }%
                }




%==============================================================
% \thanks command =============================================
%==============================================================

\let\oldFootnote\footnote

\newcommand{\EXPR@thanksMark}{\makebox[8pt]{\textsuperscript{\thefootnote}}}

\renewcommand{\thanks}[1]{%
%    \addtolength{\skip\footins}{6pt}% = 12pt = 1 line between the rule
%%
%    \renewcommand{\footnoterule}{%
%    % must have a virtual total height = 0 (cf. Latex Companion p. 72)
%    \par\vspace*{-12.3pt}% total height
%    \noindent\rule{2.5cm}{0.3pt}\vspace*{6pt} % => total height 9.3pt
%                                              % between rule and footnote text: 6pt
%    }%
%%
%    % space between each footnote
%    \setlength{\footnotesep}{18pt}%
%%
    % redefinition de \thefootnote
    \renewcommand{\thefootnote}{$\star$}%
%
    % définition de la footnote mark
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
    {%  THEN
        \providecommand{\EXPR@footnoteMark}{\unskip\kern -0.2em\mbox{\textsuperscript{\thefootnote}}}%
    }{% ELSE
        \providecommand{\EXPR@footnoteMark}{\unskip\kern -0.1em\mbox{\textsuperscript{\thefootnote}}}%
    }
%
    % redefinition of the footnote mark in the text
    \renewcommand{\@makefnmark}{\EXPR@footnoteMark}%
%
    % redefinition of the footnote itself
    \renewcommand{\@makefntext}[1]{\thefootnote.~#1}%
%
    \oldFootnote{#1}%
    % remise à zéro du compteur
    \setcounter{footnote}{0}%
}


%==============================================================
% captions ====================================================
%==============================================================


\long\def\@caption#1[#2]#3{{\setlength{\parskip}{0pt}\par\addcontentsline{\csname
   ext@#1\endcsname}{#1}{\protect\numberline{\csname
   the#1\endcsname}{\ignorespaces #2}}\begingroup
     \@parboxrestore
     \normalsize
     \@Mymakecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
   \endgroup}}

\newcommand{\@Mymakecaption}[2]{{%
    % #1 is e.g. Figure 1 or Table 1, #2 is the caption of the figure/table
   \setlength{\parskip}{0pt}%
   \par\vspace{12pt plus 2pt minus 1pt}%
   \noindent\parbox[t]{\linewidth}{\center{\normalfont\normalsize\itshape #1.}\hspace{3pt}%
                       \normalfont\normalsize\itshape #2}%
   %\vspace{1\defaultLeading} % managed by the floting objects parameters
}}


%==============================================================
% floating objects parameters ==================================
%==============================================================

\setcounter{topnumber}{2}              % déf. 2    % maximum number of fig. at the top
\setcounter{bottomnumber}{1}           % déf. 1    % maximum number of fig. at the bottom
\setcounter{totalnumber}{3}            % déf. 3    % maximum number of fig. per page
\renewcommand{\topfraction}{1}         % déf. 0.7  % maximum rate of page area filled by figures at the top
\renewcommand{\bottomfraction}{1}      % déf. 0.3  % maximum rate of page area filled by figures at the bottom
\renewcommand{\textfraction}{0}        % déf. 0.2  % minimum rate of page area filled by text
\renewcommand{\floatpagefraction}{0.8} % déf. 0.5  % floatpage fill up

% Dimensions of elasticated vertical spaces arround figures:
% play an important role for the object positioning: it is in our interrest to increase elasticity
\setlength{\floatsep}{12pt plus 2pt minus 4pt}  % additional space between 2 figures
                        % by default: \setlength{\floatsep}{12pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{20pt plus 2pt minus 8pt}% space between text and figures (at the top or the bottom of the page)
                        % by default: \setlength{\textfloatsep}{20pt plus 2pt minus 4pt}
\setlength{\intextsep}{12pt plus 2pt minus 4pt} % additional space before and after figures [h], between figure and text
                        % by default: \setlength{\intextsep}{\floatsep}


%==============================================================
% table environments ==========================================
%==============================================================

\let\@originalBeginTable\table
\let\@originalEndTable\endtable
\newsavebox{\@tableBox}

\newlength{\spaceBeforeTable}
\@ifpackageloaded{subfig}{% THEN
    \setlength{\spaceBeforeTable}{0pt}%
}{%  ELSE
    \setlength{\spaceBeforeTable}{-12pt}%
}

%%% use:
%%%     [#1]: label (optional)
%%%     #2: caption
%%%     \begin{table}[#1]{#2} ... \end{table}
\renewenvironment{table}[2][NIL]%
    {% BEGIN
     \@originalBeginTable[htbp]
     \ifthenelse{\boolean{BOOL@englishEnabled}}%
     {% THEN
        \renewcommand{\tablename}{Table}%
     }{%    ELSE
        \renewcommand{\tablename}{Tableau}%
     }

     \vspace*{\the\spaceBeforeTable}%
     \setlength{\fboxsep}{2mm}%
     \caption{#2}%
     % IF no label has been specified
        \ifthenelse{\equal{#1}{NIL}}%
     % THEN we do nothing
        {}%
     % ELSE
        {\label{#1}}%
     \medskip%
     \begin{lrbox}{\@tableBox}%
     \begin{minipage}{120mm}%
     \center%
    }
    {% END
     \end{minipage}%
     \end{lrbox}%
%     \framebox[120mm]{\usebox{\@tableBox}}%
     \usebox{\@tableBox}%
     \@originalEndTable%
    }%


%==============================================================
% macro pour référer à la table ===============================
%==============================================================

\newcommand{\tableref}[1]{\tablename~\ref{#1}}


%%%%%%==============================================================
%%%%%% algorithm environments ==========================================
%%%%%%==============================================================
%%%%%
%%%%%\ifthenelse{\boolean{BOOL@englishEnabled}}%
%%%%%{%  THEN
%%%%%    \newcommand{\algorithmname}{Algorithm}
%%%%%}{% ELSE
%%%%%    \newcommand{\algorithmname}{Algorithme}
%%%%%}
%%%%%
%%%%%% compteurs
%%%%%\newcounter{COUNT@algorithm}
%%%%%\setcounter{COUNT@algorithm}{0}
%%%%%
%%%%%\newcounter{algorithm}
%%%%%\renewcommand\thealgorithm{\arabic{COUNT@algorithm}}
%%%%%\def\fps@algorithm{tbp}
%%%%%\def\ftype@algorithm{2}
%%%%%\def\ext@algorithm{lot}
%%%%%\def\fnum@algorithm{\algorithmname\nobreakspace\thealgorithm}
%%%%%\newenvironment{algorithm}
%%%%%               {\@float{algorithm}}
%%%%%               {\end@algorithm}
%%%%%\newenvironment{algorithm*}
%%%%%               {\@dblfloat{algorithm}}
%%%%%               {\end@dblalgorithm}
%%%%%
%%%%%\let\@originalBeginAlgo\algorithm
%%%%%\let\@originalEndAlgo\endalgorithm
%%%%%
%%%%%\newsavebox{\@algoBox}
%%%%%
%%%%%%%% use:
%%%%%%%%     [#1]: label (optional)
%%%%%%%%     #2: caption
%%%%%%%%     \begin{table}[#1]{#2} ... \end{table}
%%%%%\renewenvironment{algorithm}[2][NIL]%
%%%%%    {% BEGIN
%%%%%     \@originalBeginAlgo[htbp]
%%%%%     \vspace*{-12pt}%
%%%%%     \setlength{\fboxsep}{2mm}%
%%%%%%     \refstepcounter{COUNT@algorithm}%
%%%%%%     \@Mymakecaption{\algorithmName~\arabic{COUNT@algorithm}}{#2}
%%%%%     \caption{#2}
%%%%%     % IF no label has been specified
%%%%%        \ifthenelse{\equal{#1}{NIL}}%
%%%%%     % THEN we do nothing
%%%%%        {}%
%%%%%     % ELSE
%%%%%        {\label{#1}}%
%%%%%     \medskip%
%%%%%     \begin{lrbox}{\@algoBox}%
%%%%%     \begin{minipage}{120mm}%
%%%%%     \center%
%%%%%    }
%%%%%    {% END
%%%%%     \end{minipage}%
%%%%%     \end{lrbox}%
%%%%%%     \framebox[120mm]{\usebox{\@tableBox}}%
%%%%%     \usebox{\@algoBox}%
%%%%%     \@originalEndAlgo%
%%%%%    }%
%%%%%
%%%%%
%%%%%%==============================================================
%%%%%% macro pour référer à l'algorithme ===========================
%%%%%%==============================================================
%%%%%
%%%%%\newcommand{\algorithmref}[1]{\algorithmname~\ref{#1}}


%==============================================================
% figure environments =========================================
%==============================================================

\let\@originalBeginFigure\figure
\let\@originalEndFigure\endfigure
\newsavebox{\@figureBox}
\newsavebox{\@figureCaptionBox}

% renamed traditional figure environment
%%% use:
%%%     \begin{figure}[position on the sheet]
%%%     ...
%%%     \end{figure}
\newenvironment{figuretex}{\@originalBeginFigure}{\@originalEndFigure}

% new figure environment (!!! incompatible with subfig package !!!! Then use figuretex environment)
%%% use:
%%%     \begin{figure}[caption label]{caption}
%%%     ...
%%%     \end{figure}
\renewenvironment{figure}[2][NIL]%
    {% BEGIN
     \@originalBeginFigure[htbp]
     \renewcommand{\figurename}{Figure}
     % IF no label has been specified
        \ifthenelse{\equal{#1}{NIL}}%
     % THEN we just save the caption into a box
        {\sbox\@figureCaptionBox{\caption{#2}}}%
     % ELSE we add to it the label
        {\sbox\@figureCaptionBox{\caption{#2}\label{#1}}}%
     \setlength{\fboxsep}{5mm}%
     \begin{lrbox}{\@figureBox}%
     \begin{minipage}{120mm}%
     \center%
    }%
    {% END
     \end{minipage}%
     \end{lrbox}%
%     \framebox[120mm]{\usebox{\@figureBox}}%
     \usebox{\@figureBox}%
     \par\medskip\noindent\usebox{\@figureCaptionBox}%
     \@originalEndFigure%
    }

%==============================================================
% figure encadrée environments ================================
%==============================================================


%%% use:
%%%     \begin{figure}[label]{caption}
%%%     ...
%%%     \end{figure}
\newenvironment{figureframe}[2][NIL]%
    {% BEGIN
     \@originalBeginFigure[htbp]
     \renewcommand{\figurename}{Figure}
     % IF no label has been specified
        \ifthenelse{\equal{#1}{NIL}}%
     % THEN we just save the caption into a box
        {\sbox\@figureCaptionBox{\caption{#2}}}%
     % ELSE we add to it the label
        {\sbox\@figureCaptionBox{\caption{#2}\label{#1}}}%
     \setlength{\fboxsep}{5mm}%
     \begin{lrbox}{\@figureBox}%
     \begin{minipage}{120mm}%
     \center%
    }%
    {% END
     \end{minipage}%
     \end{lrbox}%
     \framebox[120mm]{\usebox{\@figureBox}}%
%     \usebox{\@figureBox}%
     \par\medskip\noindent\usebox{\@figureCaptionBox}%
     \@originalEndFigure%
    }

%==============================================================
% macro pour référer à la figure ==============================
%==============================================================

\newcommand{\figureref}[1]{\figurename~\ref{#1}}

%==============================================================
% acknowledgements ============================================
%==============================================================

\newcommand{\acknowledgements}[1]{%
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            \newcommand{\acknowledgementsname}{Acknowledgements}%
        }{% ELSE
            \newcommand{\acknowledgementsname}{Remerciements}%
        }%
    \paragraph*{\emph{\acknowledgementsname}}
    \newlength{\ident@acknowledgements}%
    \setlength{\ident@acknowledgements}{\textwidth-0.3cm}%
    \noindent%
    \parbox{0.3cm}{\strut}%
    \parbox{\the\ident@acknowledgements}{\normalfont\normalsize\itshape #1}%
}

%==============================================================
%==============================================================
% MATHEMATICAL ENVIRONMENTS ===================================
%==============================================================
%==============================================================


%==============================================================
% counter and fonts ===========================================
%==============================================================


% -------------------------------------------------------------
% environments counter (the same for all environments)
% -------------------------------------------------------------
\newcounter{COUNT@theorem}
\setcounter{COUNT@theorem}{0}

% -------------------------------------------------------------
% theorems labels font
% -------------------------------------------------------------
\newcommand{\FONT@theoremLabel}{%
    \normalfont\normalsize\sc
}

% -------------------------------------------------------------
% font of label optionnal argument
% -------------------------------------------------------------
\newcommand{\FONT@optionnalTheoremLabel}{%
    \normalfont\normalsize
}

% -------------------------------------------------------------
% theorems body font
% -------------------------------------------------------------
\newcommand{\FONT@theoremBody}{%
    % IF the environment is boxed
    \ifthenelse{\boolean{BOOL@qedSymbolHasAValue}}%
        {% THEN no emphasis
            \normalfont\normalsize
        }{% ELSE emphase
            \normalfont\normalsize\em
        }%
}

%==============================================================
% QED symbols =================================================
%==============================================================

% -------------------------------------------------------------
% QED symbol value
% -------------------------------------------------------------
\newboolean{BOOL@qedSymbolHasAValue}         % false by default
\newcommand{\VAR@currentQedSymbolValue}{}

% -------------------------------------------------------------
% QED symbol update
% -------------------------------------------------------------
\newcommand{\SET@currentQedSymbolValue}[1]{%
    \ifthenelse{\equal{#1}{NIL}}%
        {%  THEN there is no symbol
            \renewcommand{\VAR@currentQedSymbolValue}{}%
            \setboolean{BOOL@qedSymbolHasAValue}{false}%
        }{% ELSE save the new symbol
            \renewcommand{\VAR@currentQedSymbolValue}{#1}%
            \setboolean{BOOL@qedSymbolHasAValue}{true}%
        }%
}

% -------------------------------------------------------------
% QED symbol expression
% -------------------------------------------------------------
\newcommand{\EXPR@currentQedSymbol}{%
    \ifthenelse{\boolean{BOOL@qedSymbolHasAValue}}%
        {%  THEN write symbol at the end of the current line
            {\setlength{\parskip}{0cm}\hspace*{\fill}\nolinebreak[2]\hspace*{\fill}\VAR@currentQedSymbolValue}%
        }{% ELSE do nothing
        }%
}

% -------------------------------------------------------------
% manual QED symbol call
% -------------------------------------------------------------
% The QED symbol is normally called in \end{mathematicalEnvironment}
% but if the last line begins by a command \item (enumerate, description,
% or itemize environment) the QED symbol is not at the end of this line but
% at the end of the next line.
% For this reason, the author can call the QED symbol manually, at
% the end of the line beginning by \item. In this case, the QED symbol
% must not be writen in \end{mathematicalEnvironment}

% the following boolean is true iff the command \QED has been called
\newcounter{BOOL@manualQedCalled}
\setcounter{BOOL@manualQedCalled}{0}

\newcommand{\QED}{%
    \EXPR@currentQedSymbol%
    \setcounter{BOOL@manualQedCalled}{1}
    %\par\vspace{1mm}% correction of vertical space in itemize and enumerate environment
}


%==============================================================
% definition of new environments ==============================
%==============================================================

% Several meta-command \BUID@xxx (where xxx is a mathematical environment type:
% normal, normal stared, cite, cite stared) will be defined in the following.
% Each of them have the same parameters:
% [#1]: QED symbol
% #2: environment name
% #3: environment label

% -------------------------------------------------------------
% END part of mathematical environments
% -------------------------------------------------------------
% it is the same for each environment type. Thus, we define it only one times.
\newcommand{\EXPR@endOfMathEnv}{%
    % IF the QED symbol has not been called manually
    \ifthenelse{\value{BOOL@manualQedCalled}=1}%      old version with boolean {\not\boolean{BOOL@manualQedCalled}}%
        {%  THEN no QED symbol must be printed
            \endsloppypar\par%
            % the QED has not been printed for the next environment
            \setcounter{BOOL@manualQedCalled}{0}%
        }{% ELSE we print QED symbol
            \EXPR@currentQedSymbol\endsloppypar\par%
        }%
}

% -------------------------------------------------------------
% meta-command for normal mathematical environment (with an absolute number)
% -------------------------------------------------------------
\newcommand{\BUILD@normalMathEnv}[3][NIL]{%
    % parameter of the defined mathematical environment
    % [#1]: expression that will appear between parenthesis
    \newenvironment{#2}[1][NIL]%
        {%  BEGIN
            \sloppypar%
            % set of the current box symbol
            \SET@currentQedSymbolValue{#1}%
            %
            \par\noindent\refstepcounter{COUNT@theorem}%
            % IF no optional paramater is specified for the environment
            \ifthenelse{\equal{##1}{NIL}}%
                {%  THEN
                    \mbox{\FONT@theoremLabel #3~\arabic{COUNT@theorem}.~---~}%
                }{% ELSE
                    {\FONT@theoremLabel #3~\arabic{COUNT@theorem}~\FONT@optionnalTheoremLabel(##1).~---~}%
                }%
                % we select the correct font for the body
                \FONT@theoremBody
        }{% END
            \EXPR@endOfMathEnv
        }%
}

% -------------------------------------------------------------
% meta-command for stared normal mathematical environment (without numbering)
% -------------------------------------------------------------
\newcommand{\BUILD@staredNormalMathEnv}[3][NIL]{%
    % parameter of the defined mathematical environment
    % [#1]: expression that will appear between parenthesis
    \newenvironment{#2*}[1][NIL]%
        {%  BEGIN
            \sloppypar%
            % set of the current box symbol
            \SET@currentQedSymbolValue{#1}%
            %
            \par\noindent%\refstepcounter{COUNT@theorem}%
            % IF no optional paramater is specified for the environment
            \ifthenelse{\equal{##1}{NIL}}%
                {%  THEN
                    \mbox{\FONT@theoremLabel #3.~---~}%
                }{% ELSE
                    {\FONT@theoremLabel #3~\FONT@optionnalTheoremLabel(##1).~---~}%
                }%
                % we select the correct font for the body
                \FONT@theoremBody
        }{% END
            \EXPR@endOfMathEnv
        }%
}

% -------------------------------------------------------------
% meta-command for normal mathematical environment (with an absolute number)
% having just a citation between parenthesis
% -------------------------------------------------------------
\newcommand{\BUILD@citeMathEnv}[3][NIL]{%
    % parameter of the defined mathematical environment
    % [#1]: optionnal parameter of command \cite
    % #2: parameter of command \cite
    \newenvironment{#2Cite}[2][NIL]%
        {%  BEGIN
            \sloppypar%
            % set of the current box symbol
            \SET@currentQedSymbolValue{#1}%
            %
            \par\noindent\refstepcounter{COUNT@theorem}%
            % IF no optional paramater is specified for the environment
            \ifthenelse{\equal{##1}{NIL}}%
                {%  THEN
                    \mbox{\FONT@theoremLabel #3~\arabic{COUNT@theorem}~\FONT@optionnalTheoremLabel\cite{##2}.~---~}%
                }{% ELSE
                    {\FONT@theoremLabel #3~\arabic{COUNT@theorem}~\FONT@optionnalTheoremLabel\cite[##1]{##2}.~---~}%
                }%
                % we select the correct font for the body
                \FONT@theoremBody
        }{% END
            \EXPR@endOfMathEnv
        }%
}

% -------------------------------------------------------------
% meta-command for stared mathematical environment (without numbering)
% having just a citation between parenthesis
% -------------------------------------------------------------
\newcommand{\BUILD@staredCiteMathEnv}[3][NIL]{%
    % parameter of the defined mathematical environment
    % [#1]: optionnal parameter of command \cite
    % #2: parameter of command \cite
    \newenvironment{#2Cite*}[2][NIL]%
        {%  BEGIN
            \sloppypar%
            % set of the current box symbol
            \SET@currentQedSymbolValue{#1}%
            %
            \par\noindent\refstepcounter{COUNT@theorem}%
            % IF no optional paramater is specified for the environment
            \ifthenelse{\equal{##1}{NIL}}%
                {%  THEN
                    \mbox{\FONT@theoremLabel #3~\FONT@optionnalTheoremLabel\cite{##2}.~---~}%
                }{% ELSE
                    {\FONT@theoremLabel #3~\FONT@optionnalTheoremLabel\cite[##1]{##2}.~---~}%
                }%
                % we select the correct font for the body
                \FONT@theoremBody
        }{% END
            \EXPR@endOfMathEnv
        }%
}


%==============================================================
% predefinited environments ===================================
%==============================================================

% -------------------------------------------------------------
% creation of new mathematical environment
% -------------------------------------------------------------
\renewcommand{\newtheorem}[3][NIL]{%
    % some mathematical environment may already be defined
    % when it is the case, we cancel the old definition
    \ifx\csname #2\endcsname\undefined
    \else
        \expandafter\let\csname #2\endcsname\relax
        \expandafter\let\csname end#2\endcsname\relax
        \expandafter\let\csname #2name\endcsname\relax
    \fi
    % building of environment name
    \expandafter\newcommand\csname #2name\endcsname{#3}
    % building of environment reference
    \expandafter\newcommand\csname #2ref\endcsname[1]{\csname #2name\endcsname~\ref{##1}}
    % building of a (new) definition of the environment
    \BUILD@normalMathEnv[#1]{#2}{#3}
    \BUILD@staredNormalMathEnv[#1]{#2}{#3}
    \BUILD@citeMathEnv[#1]{#2}{#3}
    \BUILD@staredCiteMathEnv[#1]{#2}{#3}
}

% -------------------------------------------------------------
% instances of predefined mathematical environments
% -------------------------------------------------------------

\ifthenelse{\boolean{BOOL@englishEnabled}}%
{%  THEN
    \newtheorem[$\square$]{assumption}{Assumption}
    \newtheorem{axiom}{Axiom}
    \newtheorem{axioms}{Axioms}
    \newtheorem[$\square$]{case}{Case}
    \newtheorem[$\square$]{claim}{Claim}
    \newtheorem[$\square$]{conclusion}{Conclusion}
    \newtheorem[$\square$]{conclusions}{Conclusions}
    \newtheorem[$\square$]{condition}{Condition}
    \newtheorem[$\square$]{conjecture}{Conjecture}
    \newtheorem{corollary}{Corollary}
    \newtheorem[$\square$]{criterion}{Criterion}
    \newtheorem{definition}{Definition}
    \newtheorem{definitions}{Definitions}
    \newtheorem[$\square$]{example}{Example}
    \newtheorem[$\square$]{examples}{Examples}
    \newtheorem[$\square$]{exercise}{Exercise}
    \newtheorem[$\square$]{fact}{Fact}
    \newtheorem[$\square$]{facts}{Facts}
    \newtheorem{hypothesis}{Hypothesis}
    \newtheorem{lemma}{Lemma}
    \newtheorem{notation}{Notation}
    \newtheorem{notations}{Notations}
    \newtheorem[$\square$]{note}{Note}
    \newtheorem[$\square$]{observation}{Observation}
    \newtheorem[$\square$]{observations}{Observations}
    \newtheorem[$\square$]{principle}{Principle}
    \newtheorem[$\square$]{problem}{Problem}
    \newtheorem[$\square$]{problems}{Problems}
    \newtheorem[$\blacksquare$]{proof}{Proof}
    \newtheorem[$\blacksquare$]{proofsketch}{Proof sketch}
    \newtheorem{property}{Property}
    \newtheorem{proposition}{Proposition}
    \newtheorem[$\square$]{question}{Question}
    \newtheorem[$\square$]{remark}{Remark}
    \newtheorem[$\square$]{remarks}{Remarks}
    \newtheorem[$\square$]{solution}{Solution}
    \newtheorem[$\square$]{summary}{Summary}
    \newtheorem{theorem}{Theorem}
}{% ELSE
    \newtheorem[$\square$]{assumption}{Supposition}
    \newtheorem{axiom}{Axiome}
    \newtheorem{axioms}{Axiomes}
    \newtheorem[$\square$]{case}{Cas}
    \newtheorem[$\square$]{claim}{Déclaration}
    \newtheorem[$\square$]{conclusion}{Conclusion}
    \newtheorem[$\square$]{conclusions}{Conclusions}
    \newtheorem[$\square$]{condition}{Condition}
    \newtheorem[$\square$]{conjecture}{Conjecture}
    \newtheorem{corollary}{Corollaire}
    \newtheorem[$\square$]{criterion}{Critère}
    \newtheorem{definition}{Définition}
    \newtheorem{definitions}{Définitions}
    \newtheorem[$\square$]{example}{Exemple}
    \newtheorem[$\square$]{examples}{Exemples}
    \newtheorem[$\square$]{exercise}{Exercice}
    \newtheorem[$\square$]{fact}{Fait}
    \newtheorem[$\square$]{facts}{Faits}
    \newtheorem{hypothesis}{Hypothèse}
    \newtheorem{lemma}{Lemme}
    \newtheorem{notation}{Notation}
    \newtheorem{notations}{Notations}
    \newtheorem[$\square$]{note}{Note}
    \newtheorem[$\square$]{observation}{Observation}
    \newtheorem[$\square$]{observations}{Observations}
    \newtheorem[$\square$]{principle}{Principe}
    \newtheorem[$\square$]{problem}{Problème}
    \newtheorem[$\square$]{problems}{Problèmes}
    \newtheorem[$\blacksquare$]{proof}{Preuve}
    \newtheorem[$\blacksquare$]{proofsketch}{Amorce de preuve}
    \newtheorem{property}{Propriété}
    \newtheorem{proposition}{Proposition}
    \newtheorem[$\square$]{question}{Question}
    \newtheorem[$\square$]{remark}{Remarque}
    \newtheorem[$\square$]{remarks}{Remarques}
    \newtheorem[$\square$]{solution}{Solution}
    \newtheorem[$\square$]{summary}{Résumé}
    \newtheorem{theorem}{Théorème}
}


%==============================================================
%==============================================================
% BIBLIOGRAPHY ================================================
%==============================================================
%==============================================================


% The following commands are apacite.sty customization commands
% see apacite.pdf (documentation of apacite style)
% Some commands are redefined during the apacite language
% customization (french.apc or english.apc files). Thus, theses
% commands must be customized at begin document. (And we thus use
% the \AtBeginDocument command to made that.)

%==============================================================
% bibliography style =========================================
%==============================================================

\bibliographystyle{\filespath hermes-journal}

%==============================================================
% bibliography layout =========================================
%==============================================================

% Except the first line, the other ones begin at 0.5cm from left margin
\setlength{\bibleftmargin}{0.5cm}
\setlength{\bibindent}{-0.5cm}

% vertical space between two references must be 6pt
\renewcommand{\bibitemsep}{6pt plus 1pt minus 1pt}

%==============================================================
% citation commands ===========================================
%==============================================================

\renewcommand{\cite}{%
    \def\BCAY##1##2##3{\BCA{##2}{##2}}% kept for compat. with prev. versions
    \def\citeauthoryear##1##2##3{\BCA{##2}{##2}}%
    \def\@BBOP{\BBOP}% % open parenthesis
    \def\@BBCP{\BBCP}% % close parenthesis
    \@ifnextchar<%     % >
        {\@cite}%
        {\@cite<>}%
}
\renewcommand{\fullcite}{\cite}
\renewcommand{\shortcite}{\cite}
\renewcommand{\citeNP}{\cite}
\renewcommand{\fullciteNP}{\cite}
\renewcommand{\shortciteNP}{\cite}
%\renewcommand{\citeA}{\cite}
\renewcommand{\fullciteA}{\cite}
\renewcommand{\shortciteA}{\cite}

%==============================================================
% global typography ===========================================
%==============================================================

% URL style definition: we want roman font
\AtBeginDocument{\urlstyle{APACrm}}

% font size for bibliography: we want size = 9pt
\renewcommand{\bibliographytypesize}{\normalfont\fontsize{9}{10.8}\selectfont}

% we don't want any string before url print
\AtBeginDocument{\renewcommand{\BAvailFrom}{\unskip}}

\AtBeginDocument{
   \renewcommand{\BOthers}{\em et al}
   \renewcommand{\BOthersPeriod}{\em et al}
}

\AtBeginDocument{
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
        {%  THEN
            % do nothing
        }{% ELSE
            \renewcommand{\refname}{Bibliographie}% Name of ref. list if it's a section.
        }
}

%==============================================================
% global punctuation ==========================================
%==============================================================

\AtBeginDocument{
   \renewcommand{\BBAA}{\unskip,}
%   \renewcommand{\BBAB}{\unskip,}
   \renewcommand{\BCBT}{}
   \renewcommand{\BCBL}{}
}

%==============================================================
% language-dependent punctuation customization ================
%==============================================================

\AtBeginDocument{%
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
    {%  THEN: english typography (default values of apacite.sty)
       % \renewcommand{\BVOL}{}
       % \renewcommand{\BVOLS}{}
       % \renewcommand{\BNUM}{}
       % \renewcommand{\BNUMS}{}
       % \renewcommand{\BPGS}{}
       \renewcommand{\BBYY}{; }
    }{% ELSE: french typography
       \renewcommand{\BVOL}{vol.}
       \renewcommand{\BVOLS}{vols.}
       \renewcommand{\BNUM}{n\up{o}}
       \renewcommand{\BNUMS}{n\up{o}}
       \renewcommand{\BPGS}{\BPG}
       \renewcommand{\BBYY}{~; }
    }%
}

%==============================================================
% labels customization ========================================
%==============================================================

\AtBeginDocument{%
    \ifthenelse{\boolean{BOOL@englishEnabled}}%
    {%  THEN: english typography (default values of apacite.sty)
        \renewcommand{\BTR}{Technical report}
    }{% ELSE: french typography
        % nothing
    }
}
%==============================================================
% journal customization =======================================
%==============================================================

% we don't want the `volume (number)' form, but `vol. x,

\AtBeginDocument{%
   \renewcommand{\APACjournalVolNumPages}[4]{%
     \Bem{#1}%             journal
     \ifx\@empty#2\@empty
     \else
       \unskip, \BVOL~{#2}%  volume
     \fi
     \ifx\@empty#3\@empty
     \else
       \unskip, \BNUM~{#3}%      issue number
     \fi
     \ifx\@empty#4\@empty
     \else
       \unskip, \BPGS~{#4}%      pages
     \fi
   }%
}

%==============================================================
% book, proceedings and collection customization ==============
%==============================================================

% We don't want `:' between the publisher and its address but `, '

\AtBeginDocument{%
   \renewcommand{\APACaddressPublisher}[2]{%
     \ifx\@empty#1\@empty
       \ifx\@empty#2\@empty
       \else
         \unskip{#2}%          publisher
       \fi
     \else
       {#1}%                   address
       \ifx\@empty#2\@empty
       \else
         \unskip, {#2}%        publisher
       \fi
     \fi
   }
}

%==============================================================
% report customization ========================================
%==============================================================

% -------------------------------------------------------------
% We don't want the string `:' between the publisher and its
% address but the string `, '
% -------------------------------------------------------------

\AtBeginDocument{\let\APACaddressInstitution\APACaddressPublisher}


% -------------------------------------------------------------
% we dropt parenthesizes arround report number, volume, edition
% and add a dot before type.
% -------------------------------------------------------------

\AtBeginDocument{%
   \renewcommand{\APACbVolEdTR}[2]{%
     \ifx\@empty#1\@empty
       \ifx\@empty#2\@empty
       \else%
        \unskip. {#2}%            (Technical Report No.\ <no>)
       \fi
     \else
       \ifx\@empty#2\@empty
       \else
         \unskip. {#2}%     ; Technical Report No.\ <no>
       \fi
       \unskip, {#1}%               (2nd ed., Vol.~1)
     \fi
   }%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\endinput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
