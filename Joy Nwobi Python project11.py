# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:50:46 2024

@author: Staff
"""


pip install textblob
import nltk
from nltk.corpus import gutenberg
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import string
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
# Load text
text = gutenberg.raw('austen-sense.txt') 

# Lowercase
text = text.lower()

# Punctuation removal
text = "".join([char for char in text if char not in string.punctuation])


# Tokenization
tokens = word_tokenize(text) 

# Get word count
word_count = len(tokens)


# POS Tagging
pos_tags = pos_tag(tokens)


# Word length stats
word_lengths = [len(word) for word in tokens]
avg_word_length = sum(word_lengths) / len(word_lengths)
print("Average word length:", avg_word_length) 

# POS tag counts  
from collections import Counter
pos_counts = Counter(tag for word, tag in pos_tags)
print("Part of speech tag counts:", pos_counts)

tag_counts = Counter(tag for word, tag in pos_tags)

print(f"Noun count: {tag_counts['NN']}") 
print(f"Adjective count: {tag_counts['JJ']}")
print(f"Verb count: {tag_counts['VB']}") 
print(f"Adverb count: {tag_counts['RB']}")

functional_tags = ['DT','PR','WP','WRB']
functional_count = 0
for tag in functional_tags:
    functional_count += tag_counts[tag]
print(f"Functional word count: {functional_count}")



# Stopwords removal  
stop_words = set(stopwords.words('english'))
words = [w for w in tokens if not w in stop_words]



# Stemming
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words] 



# Lemmatization
wordlemmatizer = WordNetLemmatizer()
lemmatized = [wordlemmatizer.lemmatize(word) for word in words]



# N-grams
text = gutenberg.raw('austen-sense.txt')
documents = sent_tokenize(text) 

all_words = " ".join([text for text in documents]).split()
bigrams = ngrams(all_words, 2)

counts = Counter(bigrams)
print(counts)



# Word clouds 
wordcloud = WordCloud().generate(' '.join(words))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



# TF-IDF
# Load text
text = gutenberg.raw('austen-sense.txt') 

# Tokenize text 
documents = sent_tokenize(text)

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform 
vectorizer.fit(documents)  

# Print vocabulary
print(vectorizer.vocabulary_)

# Print inverse document frequencies  
print(vectorizer.idf_)

# Alternatively print TF-IDF matrix
X = vectorizer.transform(documents).toarray()
print(X)




# Topic modeling
lda = LatentDirichletAllocation(n_components=10)
lda.fit(X)
print(lda.components_)


# Apply LDA
lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda.fit(X)

# Print topics and keywords
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx+1}:")
    for i in topic.argsort()[:-10 - 1:-1]:
        print(f"{feature_names[i]}: {topic[i]}")



# Information extraction
named_entities = nltk.ne_chunk(pos_tags)

import nltk

# Load the text of "Sense"
sense_text = nltk.corpus.gutenberg.raw('austen-sense.txt')

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(sense_text)

# Process each sentence
named_entities = []
for sentence in sentences:
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)

    # Extract named entities
    entities = nltk.ne_chunk(pos_tags)

    # Append named entities to the list
    named_entities.extend(entities)

# Filter and extract only the named entities that correspond to characters
characters = []
for entity in named_entities:
    if hasattr(entity, 'label') and entity.label() == 'PERSON':
        characters.append(' '.join([leaf[0] for leaf in entity.leaves()]))

# Print the extracted characters
print(characters)




# Text similarity
cosine = cosine_similarity(X,X)
print(cosine[:5, :5])

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the texts
nltk.download('gutenberg')
sense = nltk.corpus.gutenberg.raw('austen-sense.txt')
persuasion = nltk.corpus.gutenberg.raw('austen-persuasion.txt')

# Preprocess the texts
sense = sense.lower()
persuasion = persuasion.lower()

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the vectorizer on the texts
tfidf_matrix = vectorizer.fit_transform([sense, persuasion])

# Calculate the cosine similarity between the texts
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Print the similarity score
print(f"Similarity score: {similarity_score}")





# Sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load text from Gutenberg corpus
text = gutenberg.raw('austen-sense.txt')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the text
sentiment_scores = sid.polarity_scores(text)

# Print the sentiment scores
print("Sentiment Scores:")
for key, value in sentiment_scores.items():
    print(f"{key}: {value}")