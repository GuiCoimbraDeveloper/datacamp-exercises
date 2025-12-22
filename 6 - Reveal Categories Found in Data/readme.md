Projeto Premium
Reveal Categories Found in Data
Apply clustering and NLP techniques to sort text data from negative reviews in the Google Play Store into categories.

Descrição do Projeto
App publishers want to know what users think and feel about their apps. While we can't read their minds, you can data-mine the reviews they leave to uncover the main topics!

In this project, you will use clustering and NLP techniques to sort text data from negative reviews in the Google Play Store into categories.

To reveal the main topics from app reviews, you'll perform these tasks:

Preprocess the negative reviews (reviews with a score of 1 or 2) by tokenizing the text, removing stop words and non-alpha characters. Save the results in a pandas DataFrame called preprocessed_reviews.
Vectorize the cleaned negative reviews using TF-IDF and store the matrix in a variable called tfidf_matrix.
Apply K-means clustering to tfidf_matrix to group the reviews into five categories. Store the predicted labels in a list called categories.
For each unique cluster label, find the most frequent term. Store the results in a pandas DataFrame called topic_terms with at least three columns to store the label assigned from K-means, the identified term, and its frequency.



https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/regular-expressions-word-tokenization?ex=4

https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=4

https://www.datacamp.com/tutorial/k-means-clustering-python

https://www.datacamp.com/tutorial/text-analytics-beginners-nltk
