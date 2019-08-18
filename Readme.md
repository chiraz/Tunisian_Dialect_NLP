# Introduction and Objectives

In a nutshell, this is an open-ended project, that started out as a pet project, and aims to develop NLP resources 
and tools for the automated processing of Tunisian dialect text. 

NLP and text mining tasks require one or more forms of linguistic resources/knowledge:

- *lexical and morphological knowledge*: knowledge about individual words -- what constitutes a word in the language, how to spearate/extract words in running text (to be able to perform tokenization); how words are related to each other (for stemming and lemmatization).

- *syntactic knowledge*: knowledge about the grammatical structure of sentences, how words can be formed into correct sentences
 
- *semantic knowledge*: knowledge about meaning, how meaning of a text is related to its words and sentence structure. Sentiment lexicons are a specialized type of this knowledge.

NLP practitioners who work with the English and Spanish languages (for example) often take it for granted that sources of such knowledge are highly standardized and readily available. 
But this is not always the case. Doing NLP with under-resourced languages is challenging, and one often has to make do with heuristics and construct adhoc linguistic resources on the fly, often re-inventing the wheel. 

I found myself exactly in this situation when working with social media data in the Tunisian dialect. 

This is an open-ended pet project with the general aim of creating public open-source tools and resources for automated computational processing of Tunisian dialect text, 
including a large lexicon of common words and relationships between them, and methods for dialect detection, tokenization, morphological analysis (word segmentation), and lemmatization. 

The overarching goal of this project is really to encourage pooling and sharing of NLP resources for the Tunisian dialect as an under-resourced language.


# Tentative Plan

I have divided the project into manageable computational tasks as shown below. Each task is contained in a separate .ipynb file. 
As of July 26, 2019, only the first two tasks are available.

This is only a preliminary plan. I will be continuously updating and extending it as I gain a better, deeper understanding of the problems at stake.


1. **Part 1**: Prepare a clean labelled corpus of context-free Arabic and Tunisian dialect words.
2. **Part 2**: Build a word-level language classifier for MSA vs. Tunisian dialect.
3. **Part 3**: Build a sentence-level language classifier for MSA vs. Tunisian dialect based on the word-level classifier.
4. Generate a lexicon (wordnet) for the Tunisian dialect based on above language classifiers.
5. Induce morphological rules and a morphological analysis method for the Tunisian dialect based on above results.



# Requirements

- Python 3.6.4
- NLTK 3.4

# Contact Information

- LinkedIn: https://www.linkedin.com/in/chiraz-benabdelkader/
- Email: chiraz.benabdelkader@gmail.com
