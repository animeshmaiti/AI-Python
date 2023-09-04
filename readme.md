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
