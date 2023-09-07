# Machine learning
All simplified explain with ex.
## CoutVectorizer :

Suppose you have a collection of three text documents:

Document 1: "I like cats and dogs."
Document 2: "Dogs are friendly animals."
Document 3: "Cats and dogs make good pets."
Now, let's use `CountVectorizer` to represent these documents as a sparse matrix:

### 1. `Vocabulary Building`:
We create a vocabulary of unique words from all the documents. In this case, our vocabulary might include words like "I," "like," "cats," "and," "dogs," "are," "friendly," "animals," "make," "good," and "pets."

### 2. `Counting`:
For each document, we count how many times each word from the vocabulary appears in that document.

For example:<br>
Document 1: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] (Counts for each word in the vocabulary)<br>
Document 2: [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]<br>
Document 3: [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]<br>
### 3. `Sparse Matrix`:
The result is a matrix where each row corresponds to a document, and each column corresponds to a word from the vocabulary. The numbers in the matrix represent the counts of each word in the respective documents. However, most of these counts are zero because not all words from the vocabulary appear in every document.<br>
| 1 1 1 1 1 0 0 0 0 0 0 |<br>
| 0 0 0 1 1 1 1 1 0 0 0 |<br>
| 0 0 1 1 1 0 0 0 1 1 1 |<br>

## confusion_matrix
`The confusion_matrix(y_test, y_pred)` function is used to compute a confusion matrix for a classification problem. A confusion matrix is a table that summarizes the performance of a classification algorithm by comparing the actual (`true`) class labels (`y_test`) with the predicted class labels (`y_pred`).<br>
Here's how the confusion matrix is structured:<br>

|  |   `Predicted Negative` 	| `Predicted Positive` |
|:---:|:---:|:---:|
|`Actual Negative `|   True Negative  |    False Positive|
|`Actual Positive`  |  False Negative  |   True Positive|

1. `True Negative (TN)`: The number of instances that were correctly predicted as negative.

1. `False Positive (FP)`: The number of instances that were incorrectly predicted as positive when they were actually negative. Also known as Type I error.

1. `False Negative (FN)`: The number of instances that were incorrectly predicted as negative when they were actually positive. Also known as Type II error.

1. `True Positive (TP)`: The number of instances that were correctly predicted as positive.

