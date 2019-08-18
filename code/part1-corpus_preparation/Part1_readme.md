## Overview of Part 1 of Project

Our goal is to obtain a well-curated and fairly large list of distinct words from the MSA vocabulary, 
and a similar list for the Tunisian language (dialect). 
We will be using these word lists in subsequent parts of the project to train models that identify the language of new/unseen words.

Currently we use words obtained from the following three different sources:

1. Open Source Arabic Corpus (OSAC). Consists of 4,102,134 tokens derived from BBC Arabic and CNN Arabic websites.

2. Arabic Wordnet corpus, available in the NLTK library.

3. Words automatically extracted from a large corpus of ~ 20K Tunisian social media comments. 


The words in the first 2 corpora are mostly MSA words, but do contain named entities, Arabic dialect words, and noise (e.g. errors, misspelled words).

The words in the third corpus are manually labelled with the appropriate language label: MSA, Tunisian dialect, named entity, or noise. Some words have multiple labels, for e.g. a word could be both a named entity and an MSA word.


We will first clean up each corpus as much as possible via manual and semi-automatic methods (such as regular expressions). 
And then we will combine them into a single list of words with corresponding language labels ("MSA" for MSA Arabic words and "TND" for Tunisian dialect words).



## References and Links

### Open Source Arabic Corpora (OSAC)

- Motaz K. Saad and Wesam Ashour, "OSAC: Open Source Arabic Corpus", 6th ArchEng International Symposiums, EEECS’10 the 6th International Symposium on Electrical and Electronics Engineering and Computer Science, European University of Lefke, Cyprus, 2010.
- https://sourceforge.net/projects/ar-text-mining/files/Arabic-Corpora/
- https://sites.google.com/a/aucegypt.edu/infoguistics/directory/Corpus-Linguistics/arabic-corpora


### Arabic Wordnet Corpus (AWN)

