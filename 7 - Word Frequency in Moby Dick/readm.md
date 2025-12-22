Projeto Premium
Word Frequency in Moby Dick
Use web scraping and NLP to find the most frequent words in classic literature: Herman Melville's novel, Moby Dick.

Descrição do Projeto
In this project, you'll scrape a novel from an html file taken from the website Project Gutenberg (which contains a large corpus of books) using the Python requests package. Then you'll extract text from this web data using the BeautifulSoup library. Finally, you'll analyze the distribution of words using the Natural Language ToolKit (nltk), a very common NLP task used to gain insights on textual data and lay the groundwork for deeper analysis.

The natural language processing tools used here are required for many important data science tasks as a vast proportion of the world's data is unstructured and includes a great deal of text. To complete this project, you need to know how to import web data into Python and how to work with natural language text.

What are the most frequent words in Herman Melville's novel Moby Dick, and how often do they occur?

Note that the HTML file you are asked to request is a cashed version of this file from Project Gutenberg.

Your project will follow these steps:

The first step will be to request the Moby Dick HTML file using requests and encoding it to utf-8. Here is the URL to scrape from: https://s3.amazonaws.com/assets.datacamp.com/production/project_147/datasets/2701-h.htm
Next, you'll extract the HTML and create a BeautifulSoup object using an HTML parser to get the text.
Following that, you'll initialize a regex tokenizer object tokenizer using nltk.tokenize.RegexpTokenizer to keep only alphanumeric text, assigning the results to tokens.
You'll transform the tokens into lowercase, removing English stop words, and saving the results to words_no_stop.
Finally, you'll initialize a Counter object and find the ten most common words, saving the result to top_ten and printing to see what they are.