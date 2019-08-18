## Overview of Part 2 of Project

In this part of the project, our goal is to perform context-free word-level language identification of MSA Arabic vs. Tunisian dialect, 
which are considered very similar languages. 
This is meant as a first step towards ultimately building a sentence-level classifier for Arabic vs. Tunisian dialect.

We will explore two major distinct approaches to this problem: 

1. an anomaly detection approach based on a character-level language model of MSA Arabic words. 

2. a supervised learning (classification) approach where we represent words in vector space as bags of character ngrams, 
and build the language classifier using a labelled training set of words and classical supervised learning techniques,
namely logistic regression, multinomial naive bayes, random forests, and XGboost.


### Anomaly detection approach with language modeling

We first build (train) a character-level ngram language model for MSA Arabic words, which we then use to build a word classifier
that separates (classifies) words into 2 categories: Tunisian dialect and MSA. 

*What is the use of an Arabic vs. Tunisian dialect language classifier, and why do it at the word-level rather than directly at the sentence or even document level?*

The training data for our character-level ngram language model consists of a list of distinct *context-free* (standalone) Arabic words. 
These words are incorporated into the model independently of each other, and their order in the training data is irrelevant. 

Intuitively this language model partially captures **sub-word** information, i.e. the **morphology** of the Arabic language (or word formation rules), for instance the fact that certain prefixes are very common while others are unlikely. On the other hand, this model does not capture any knowledge about Arabic sentence structure (syntax rules) and word context. In other words, it does not capture *inter-word* relationships.

Once trained, we use the language model to calculate the perplexity score of an arbitrary word, and thereby to *classify* it as an MSA word or not by thresholding the perplexity score of the word. 

This is basically an **anomaly detection** approach to our dialect word recognition problem: dialectal words are treated as anomalies with respect to the language model of MSA words -- they are expected to have extreme perplexity values in this model. 

We evaluate the performance of this language classifier using a list of MSA and Tunisian dialect words.


### Supervised learning approach

1. three classical supervised learning techniques without class balancing: **logistic regression**, **Naive Bayes**, and **random forests**. For comparison purposes, we also 
build the **random classifier** that equaly andomly assigns a word to MSA or TND (i.e. like an unbiased coin toss).

2. extension above methods to deal with class imbalance in the data via downsampling and sensemble techniques.

3. XGBoost method.




## Summary of Performance Results

Since the two classes are quite **imbalanced** in our training data (around 600 TND words vs. 35000 MSA words), 
we use the **average F1 score** as a performance measure. 


### Anomaly detection approach results

- The classification performance on the *training set* is **average F1 score=0.60**, with an F1 score of 0.99 for the MSA class, and 0.22 for the TND class.

- The classification performance on the *test set* is **average F1 score=0.53**, with an F1 score of 0.97 for the MSA class, and 0.08 for the TND class.
  Also, the area under the ROC curve (AUC) is 0.69.

- These results are all based on an order of `N = 3` of the ngram language model. We did try a few other values (N= 2,4,5) but classification performance was significantly worse in each case.

- The poor performance for the TND class is due to the inherent large overlap between the perplexity score distributions for MSA and TND words.

- From a machine learning point of view, this method has **high bias**.

- One inherent limitation of this approach is that the ngrams of a word are all *weighted (i.e. contribute) equally* in the calculation of the perplexity score of the word, which 

- A sensible alternative approach worth exploring would be to use machine learning (supervised learning) with the set of ngrams corresponding to each word 
being the word's features. This approach automatically/implicitly learns appropriate feature weights that optimize classification performance.


### Supervised learning approach results


#### Feature representation and selection

- Based on qualitative visual inspection of the words containing top features, the **Chi-squared statistic** seems to be better than mutual information at discerning important (discriminant) features. 

- The number of selected top features was chosen to be $K=3000$, based on visually inspecting the ECDF of chi-squared scores.


#### Performance of baseline approaches

- Overall, all three models outperform the random classifier, with Logistic Regression slightly edging out the other two methods (Naive Bayes and Random Forests)
at an AUC of 0.85 and an average F1 score of 0.73, 0.45 for the TD class and 0.95 for the MSA class.

- Performance is quite **biased** towards the MSA class since the training set contains about 60 times MSA samples as TD samples 
(around 600 TND words vs. 35000 MSA words).

- In this part, the **data imbalance** problem is only handled in the decision function by calibrating the probability threshold value.


TO DO: present these results below in a table instead

*Random Classifier*

- AUC: Not applicable
- Average F1 score is 0.345, F1 of MSA is 0.66, F1 score of TND is 0.03

*Logistic Regression model (with default hyperparameters)*

- AUC = **0.910**
- With threshold=0.5: Average F1 score is 0.56, F1 of MSA is 0.99, of TND is 0.04
- With threshold that maximizes average F1 score: Average F1 score is **0.68**, F1 of MSA is 0.99, F1 of TND is 0.38

*Naive Bayes model (with default hyperparameters)*

- AUC = **0.885**
- With threshold=0.5: Average F1 score is 0.49, F1 score of MSA is 0.99, of TND is 0.00
- With threshold that maximizes average F1 score: Average F1 score is 0.65, F1 score of MSA is 0.99, of TND is 0.32

*Random Forests model (with n_estimators=400, max_depth=None and other default hyperparameters)*

- AUC = **0.887**
- With threshold=0.5: Average F1 score is 0.54, F1 score of MSA is 0.99, of TND is 0.09
- With threshold that maximizes average F1 score: Average is 0.66, F1 score of MSA is 0.99, of TND is 0.33




#### Performance with data imbalance handling

*BalancedBaggingClassifier with Logistic Regression as base estimator*

- AUC = **0.846**
- Average F1 score is 0.71, F1 score of MSA is 0.97, of TND is 0.45


*BalancedBaggingClassifier with Decision Tree as base estimator*

- AUC = **0.857**
- Average F1 score is 0.72, F1 score of MSA is 0.98, of TND is 0.46


*Forest of randomized trees*

- AUC = **0.839**
- Average F1 score is 0.72, F1 score of MSA is 0.975, of TND is 0.465



#### Performance of Xgboost method

TO DO ...



#### Feature importance

Based on visual inspection of the plots of feature importance scores vs. chi2 scores (from the automated feature selection section), 
as well as qualitative analysis of the top 10 features in each model, it seems that the Logistic Regression model is doing a better job at 
selecting features that discriminate between the two classes.



## Final Remarks and Conclusions


TO DO ...



## References

### Bibliographic References

- *Hierarchical Character-Word Models for Language Identification*, Aaron Jaech et al., 2016, https://aclweb.org/anthology/W16-6212
- *Word Level Language Identification in Online Multilingual Communication*, D. Nguyen and A. S. Dogruoz, 2013, https://www.aclweb.org/anthology/D13-1084
- *Token Level Identification of Linguistic Code Switching*, Heba ElFardy and Mona Diab, 2012.

### Programming References

- https://www.nltk.org/api/nltk.lm.html
- https://www.nltk.org/_modules/nltk/test/unit/lm/test_models.html
- http://computational-linguistics-class.org/assignment5.html
- https://www.kaggle.com/alvations/n-gram-language-model-with-nltk
- https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
