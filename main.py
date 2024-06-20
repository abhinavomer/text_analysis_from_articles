# extracting positive words
positive=open('positive-words.txt')
data=positive.read()
positive_words=data.split('\n')
positive_words.remove('')
positive.close()

# extracting negative words
negative=open('negative-words.txt')
data=negative.read()
negative_words=data.split('\n')
negative_words.remove('')
negative.close()

# extracting all stop words
stop1=open('StopWords_Auditor.txt')
data=stop1.read()
stop_words1=data.split('\n')
stop_words1.remove('')
stop1.close()

stop2=open('StopWords_Currencies.txt')
data=stop2.read()
stop_words2=data.split('\n')
stop_words2.remove('')
stop_curr=[]
for x in stop_words2:
    a=x.split("|")
    stop_curr=stop_curr+a
stop2.close()

stop3=open('StopWords_DatesandNumbers.txt')
data=stop3.read()
stop_words3=data.split('\n')
stop3.close()

stop4=open('StopWords_Generic.txt')
data=stop4.read()
stop_words4=data.split('\n')
stop4.close()

stop5=open('StopWords_GenericLong.txt')
data=stop5.read()
stop_words5=data.split('\n')
stop5.close()

stop6=open('StopWords_Geographic.txt')
data=stop6.read()
stop_words6=data.split('\n')
stop_words6.remove('')
stop6.close()

stop7=open('StopWords_Names.txt')
data=stop7.read()
stop_words7=data.split('\n')
stop_words7.remove('')
stop7.close()

# combining all stopwords
stopwords=stop_words1+stop_curr+stop_words3+stop_words4+stop_words5+stop_words6+stop_words7
len(stopwords)

# importing libraries
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

# reading data extracted file
df=pd.read_csv('article.csv',encoding='ISO-8859-1')

# performing stemming
port_stem=PorterStemmer()
def stemming(content):
    content=str(content)
    stemmed_content = content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
df['stemmed_content']=df['description'].apply(stemming)

# text analysis

def count_complex_words(words):
    a=(len(words)/(sum(1 for word in words if count_syllables(word) > 0)))
    return sum(1 for word in words if count_syllables(word) > 2),a
def count_syllables(word):
    word = word.lower()
    syllables = 0
    vowels = "aeiou"
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllables += 1
    if word.endswith("es") or word.endswith("ed"):
        syllables -= 1
    if syllables == 0:
        syllables += 1
    return syllables
def average_word_length(words):
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words)
avg_sen_len=[]
per_complex_word=[]
fog_index=[]
avg_no_words_per_sen=[]
com_word_count=[]
word_count=[]
syll_per_word=[]
personal_pronoun=[]
avg_word_count=[]
pos=[]
neg=[]
pol=[]
sub=[]

def preprocess_article(article):
    article=str(article)
    words = article.split()
    return words
def classify_article(article):
    words = preprocess_article(article)
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_count - negative_count) / ((positive_count + negative_count) + 0.000001)
    subjectivity_score = (positive_count + negative_count) / (len(words) + 0.000001)
    pos.append(positive_count)
    neg.append(negative_count)
    pol.append(polarity_score)
    sub.append(subjectivity_score)
    
    content=str(article)
    sentences = sent_tokenize(content)
    words = word_tokenize(content)
    avg_word_length=average_word_length(words)
    num_sentences = len(sentences)
    num_words = len(words)
    num_complex_words,syl_per_words = count_complex_words(words)
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', content, re.I)
    
    pronoun_len=len(pronouns)
    avg_sentence_length = int(num_words / num_sentences)
    percentage_complex_words = (num_complex_words / num_words)*100
    fo_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_sen_len.append(num_sentences)
    per_complex_word.append(percentage_complex_words)
    fog_index.append(fo_index)
    avg_no_words_per_sen.append(avg_sentence_length)
    com_word_count.append(num_complex_words)
    word_count.append(num_words)
    syll_per_word.append(syl_per_words)
    personal_pronoun.append(pronoun_len)
    avg_word_count.append(avg_word_length)
    
    if positive_count > negative_count:
        return "Positive"
    else:
        return "Negative"
df['sentiment']=df['description'].apply(classify_article)

df['positive_score']=pos
df['negative_score']=neg
df['Polarity_score']=pol
df['Subjectivity_score']=sub

df['Avg_sen_len']=avg_sen_len
df['per_com_word']=per_complex_word
df['fog_index']=fog_index
df['avg_wrod_per_sen']=avg_no_words_per_sen
df['com_words_count']=com_word_count
df['word_count']=word_count
df['syll_per_word']=syll_per_word
df['personal_pronoun']=personal_pronoun
df['Avg_word_count']=avg_word_count

# converting dataframe to csv
df.to_csv("output.csv")



