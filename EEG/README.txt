#### important points:

sleep stage has more low freq signals, wake has high freq signals

Tried classification with 4 types of feature sets: PSD, SAX, stat features on raw data, dnn on raw data

1. balance the data (subsample or SMOTE)
2. standard scaling performs better than minmax scaling with SVM, and improves its results to those from random forest classifier
3. SVM with balanced weightes performs better than unbalanced
4. It is important to choose a smaller time window for PAA (SAX method), but not too small to increase the computation
5. SMOTE + hyperparameter optimisation performed worse than just SMOTE


https://imotions.com/blog/what-is-eeg/
https://www.physionet.org/physiobank/database/sleep-edfx/
