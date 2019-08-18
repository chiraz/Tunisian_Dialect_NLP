# Source: https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def my_f1_score(precision,recall):
    return 2 * (precision * recall) / (precision + recall)

def my_performance_report(true_labels,labels_pred,prob_pred,classes_):


    ## Performance based on default probability cutoff threshold=0.5
    # This is only useful for comparison with probability calibration results below

    print("------------------------------------")
    print("Classification report with default cutoff threshold=0.5")
    print("------------------------------------")
    print("")

    # precision and recall 
    print(metrics.classification_report(true_labels,labels_pred))

    # average F1 score of the two classes
    u = pd.DataFrame(metrics.classification_report(true_labels,labels_pred,output_dict=True))
    print("Average F1 score:", u.loc["f1-score",classes_].mean())
    print("")

    # confusion matrix
    print("Confusion matrix:")
    print(pd.crosstab(pd.Series(true_labels),pd.Series(labels_pred)))
    # alternative method
    #metrics.confusion_matrix(true_labels, labels_pred)

    print("")
    print("")


    #### AUC

    #roc_auc_score(y_true,y_score): for binary y_true, y_score is supposed to be the score of the class with greater label.
    assert(classes_[0]<classes_[1])
    print("------------------------------------")
    print("Area Under the Curve:%7.3f" %(metrics.roc_auc_score(true_labels,prob_pred[:,1])))
    print("------------------------------------")
    print()

    print("")
    print("")

 
    ## Plot Average F1 Score

    i = 0
    precision0,recall0,thresholds0 = metrics.precision_recall_curve(true_labels,prob_pred[:,i],pos_label=classes_[i])
    i = 1
    precision1,recall1,thresholds1 = metrics.precision_recall_curve(true_labels,prob_pred[:,i],pos_label=classes_[i])

    f1_score0 = my_f1_score(precision0,recall0)
    f1_score1 = my_f1_score(precision1,recall1)

    #create series containing F1 scores of class1
    u0 = pd.DataFrame(zip(1-thresholds0,f1_score0[:-1]))
    u0.columns = ['threshold',classes_[0]]
    u0 = u0.set_index('threshold')
    #create series containing F1 scores of class2
    u1 = pd.DataFrame(zip(thresholds1,f1_score1[:-1]))
    u1.columns = ['threshold',classes_[1]]
    u1 = u1.set_index('threshold')
    #merge the two series into a single data frame, and align at threshold values
    u = pd.concat([u0, u1],sort=True).sort_index()
    assert(u0.shape[0]+u1.shape[0]<=u.shape[0])
    #visualize missing values after rmerge
    ##sns.heatmap(u.reset_index().loc[:,classes_].isnull(), cbar=False)
    #fill-in missing values
    u = u.fillna(method='bfill').fillna(method='ffill')
    assert(u.shape[1]==2)
    u['Average'] = u.mean(axis=1)
    d = pd.concat([u.idxmax(),u.max()],axis=1)
    d.columns = ['Threshold','Max(F1)']
    
    print("------------------------------------")
    print('Maximum F1 Scores:')
    print("------------------------------------")
    print(d.T)

    print("")
    print("------------------------------------")
    print("")

    ax = plt.gca()
    u.plot(ax=ax)
    d.plot.scatter(x='Threshold',y='Max(F1)', s=50, color='MAGENTA', ax=ax)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title('F1 score vs. Probability threshold')
    plt.legend()
    plt.axis([0,1,0,1])
    plt.show()
           
    #plt.plot(1-thresholds0,f1_score0[:-1],label=classes_[0]+"-F1")
    #plt.plot(thresholds1,f1_score1[:-1],label=classes_[1]+"-F1")
    #plt.show()


    #### Plot Precision-Recall Curve

    plt.plot(precision0,recall0,label=classes_[0])
    plt.plot(precision1,recall1, label=classes_[1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.axis([0,1,0,1])
    plt.axis('square')
    plt.show()

    plt.plot(1-thresholds0,precision0[:-1],label=classes_[0]+"-Precision")
    plt.plot(1-thresholds0,recall0[:-1],label=classes_[0]+"-Recall")
    plt.plot(thresholds1,precision1[:-1],label=classes_[1]+"-Precision")
    plt.plot(thresholds1,recall1[:-1],label=classes_[1]+"-Recall")
    plt.xlabel('Threshold')
    plt.title('Precision and Recall vs. Probability threshold')
    plt.legend()
    plt.axis([0,1,0,1])
    plt.show()

    #### Plot ROC Curve

    i = 0
    fpr0,tpr0,thresholds0 = metrics.roc_curve(true_labels,prob_pred[:,i],pos_label=classes_[i])
    i = 1
    fpr1,tpr1,thresholds1 = metrics.roc_curve(true_labels,prob_pred[:,i],pos_label=classes_[i])


    plt.plot(fpr0,tpr0,label=classes_[0])
    plt.plot(fpr1,tpr1,label=classes_[1])
    plt.axis([0,1,0,1])
    plt.axis('square')
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    plt.plot(1-thresholds0,tpr0,label=classes_[0]+"-TPR")
    plt.plot(1-thresholds0,fpr0,label=classes_[0]+"-FPR")
    plt.plot(thresholds1,tpr1,label=classes_[1]+"-TPR")
    plt.plot(thresholds1,fpr1,label=classes_[1]+"-FPR")
    plt.xlabel('Threshold')
    plt.title('TPR and FPR vs. Probability threshold')
    plt.axis([0,1,0,1])
    plt.legend()
    plt.show()


def analyze_feature_importance(feat_imp,feature_names,train_words_df,X_train):

    assert(train_words_df.shape[0]==X_train.shape[0] and feat_imp.shape[0]==X_train.shape[1])

    feat_imp_df = pd.DataFrame(dict(importance_score=feat_imp,feature=feature_names))

    feat_imp_df.plot.hist(bins=20, logy=True)

    feat_imp_df.sort_values(by='importance_score',ascending=False, inplace=True)

    # top 10 features
    print("Top 10 Features:")
    print("Importance_score\tNgram_feature")
    for i in feat_imp_df.index[0:10]:
        ##print("\t", feat_imp_df.importance_score[i], '\t\t\t\"'+feat_imp_df.feature[i]+'\"')
        print('%11.3f\t\t\"%s\"' %(feat_imp_df.importance_score[i],feat_imp_df.feature[i]))
    print("")

    ## relationship of the top 10 features with the training words

    for i in feat_imp_df.index[0:10]:
        print("=====================")
        print("feature:", '\"'+feat_imp_df.feature[i]+'\"')
        print("")
        u = X_train[:,i].toarray().ravel()
        print("Number of training words that contain this feature:",(u>0).sum())
        print("")
        df1 = train_words_df.loc[u>0,'lang_label'].value_counts(normalize=False)
        df2 = train_words_df['lang_label'].value_counts(normalize=False)
        df3 = df1/df2*100
        df3.index = df1.index
        df4 = pd.concat([df1,df2,df3],axis=1, sort=False)
        df4.columns = ['contain this feature', 'total', '% of total']
        print("Distribution of language labels of words that contain this feature:")
        print(df4)
        print("")
        print("Example training words that contain this feature:")
        print("  MSA:",train_words_df.loc[(u>0)&(train_words_df.lang_label=="MSA"),'word'].head(10).tolist())
        print("  TND:",train_words_df.loc[(u>0)&(train_words_df.lang_label=="TND"),'word'].head(10).tolist())
        print("")