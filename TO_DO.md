
## Part 1


## manual data cleaning

- manually clean up or remove noisy words in OSAC and Wordnet corpora, because they contain many misspelled words, named entities such as named entities
(such as country or person names), dialectal words, and transliterated borrowed/foreign (non-Arabic) words.

Here are some examples:

borrowed words: 'كروماتوغرافيا'

arabic named entities:  'الحريش', 'مراكش', 'جرش', 

foreign named entities (transliterated):
'فو', 'بورتوريكو', 'طوكيو', 'الكونغو', 'ماثيو', 'بيساو', 'تساو', 'غرينتش', 'فانواتو', 'مستو', 'الناتو',

misspelled words:  'ااشتري', 'لااراديا', 'ااستدعي'

dialectal words:



- resolve language labels inconsistencies between the OSAC and Wordnet corpora on one hand, and social media corpus on the other hand.

- augment the number of Tunisian dialect words in the training data (manual labelling work!).


## automated data cleaning

- remove Arabic "tatweel" character as part of Arabic letter normalization (for e.g. 'به' <== 'بـه')

- normalize glyphs, for e.g. 'ﷲ', 'ﻻ' are one-character words. Should convert them into multi-letter words.

- convert extended arabic characters to standard arabic characters (e.g. 0x6a9 --> 0x643)



## Part 2

### anomaly detection and language modeling

- use KenLM (https://kheafield.com/code/kenlm/estimation/) instead of NLTK.lm module.
- use skipgrams in addition to ngrams.
- Analyze and interpret classification errors (false positives, false negatives, as a function of `word_len`, `in_train`, and `lang_label`)

### feature selection

- use `CountVectorizer` instead of `TfidfVectorizer`.
- optimize values of `N` and `K` for accuracy, where `N` is the maximum order of the ngram features, and `K` is the number of features in feature selection. Currently we use fixed values of N=4 and K=3000.
- use different automatic feature selection methods such as L1-based feature selection (See https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel).
- use skipgrams in addition to ngrams in the bag-of-ngrams model.


### supervised learning models

- Hyperparameter tuning (grid search with cross-validation) for logistic regression and Naive Bayes.
- Hyperparameter tuning with more hyperparameters for random forests (so far we've only tuned `n_estimators` and `max_depth`).
- semi-supervised learning