import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk

# NLTK resources

nltk.download("punkt")
nltk.download("stopwords")

# Functions to clean text using stop words lists


def clean_text(text):
    stop_words = set(stopwords.words("english"))

    # Load custom stop words

    stop_words_files = [
        "Auditor",
        "Currencies",
        "DatesandNumbers",
        "Generic",
        "GenericLong",
        "Geographic",
        "Names",
    ]

    for file in stop_words_files:
        with open(
            os.path.join("StopWords", f"StopWords_{file}.txt"), "r", encoding="utf-8"
        ) as f:
            stop_words.update(f.read().splitlines())

    # Remove stop words
    words = word_tokenize(text)
    words = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]

    return " ".join(words)


# Function to load positive and negative words from folder named "MasterDictionary"
def load_pos_neg_list(file_name):
    file_path = os.path.join("MasterDictionary", file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        word_list = file.read().splitlines()
    return set(word_list)


# Function to compute variables and peform text analysis
def analyze_text(text):
    # Tokenize text into sentences and words

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Sentiment analysis using NLTK's SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    # Load positive and negative words
    positive_words = load_pos_neg_list("positive-words.txt")
    negative_words = load_pos_neg_list("negative-words.txt")

    # Calculate positive and negative scores
    postive_score = sum([1 for word in words if word.lower() in positive_words])
    negative_score = sum([1 for word in words if word.lower() in negative_words]) * -1

    # Calculate polarity and subjectivity scores
    polarity_score = (postive_score - negative_score) / (
        (postive_score + negative_score) + 0.000001
    )
    subjectivity_score = (postive_score + negative_score) / (len(words) + 0.000001)

    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences)

    # Calculate percentage of complex words
    ps = PorterStemmer()
    complex_word_count = sum(1 for word in words if len(ps.stem(word)) > 2)
    percentage_complex_words = (complex_word_count / len(words)) * 100

    # Calculate Fog index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Calculate average number of words per sentence
    avg_words_per_sentence = len(words) / len(sentences)

    # Count complex words
    complex_words = [word for word in words if len(ps.stem(word)) > 2]

    # Count syllables per word
    syllable_per_word = sum([count_syllables(word) for word in complex_words]) / len(
        complex_words
    )

    # Count personal pronouns
    personal_pronouns = len(
        re.findall(
            r"\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves)\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Calculate average word length
    avg_word_length = sum([len(word) for word in words]) / len(words)

    return (
        pos_score,
        neg_score,
        polarity_score,
        subjectivity_score,
        avg_sentence_length,
        percentage_complex_words,
        fog_index,
        avg_words_per_sentence,
        len(complex_words),
        len(words),
        syllable_per_word,
        personal_pronouns,
        avg_word_length,
    )


# Function to count syllables per word
def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("es") or word.endswith("ed"):
        count -= 1
    if count == 0:
        count += 1
    return count


# Function to extract article text and title from URL using Selenium and BeautifulSoup
def extract_article_text(url):
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract article title and text
        article_title = soup.title.text if soup.title else ""
        article_text = ""
        paragraphs = soup.find_all("p")
        for paragraph in paragraphs:
            article_text += paragraph.text + "\n"

        return article_title, article_text

    except Exception as e:
        print(e)
        return "", ""

    finally:
        driver.quit()


# Read input Excel file
input_df = pd.read_excel("Input.xlsx")

# Initialize output DataFrame
output_columns = [
    "URL_ID",
    "URL",
    "POSITIVE SCORE",
    "NEGATIVE SCORE",
    "POLARITY SCORE",
    "SUBJECTIVITY SCORE",
    "AVG SENTENCE LENGTH",
    "PERCENTAGE OF COMPLEX WORDS",
    "FOG INDEX",
    "AVG NUMBER OF WORDS PER SENTENCE",
    "COMPLEX WORD COUNT",
    "WORD COUNT",
    "SYLLABLE PER WORD",
    "PERSONAL PRONOUNS",
    "AVG WORD LENGTH",
]

output_df = pd.DataFrame(columns=output_columns)

# Iterate through each row in the input DataFrame
for index, row in input_df.iterrows():
    url_id = row["URL_ID"]
    url = row["URL"]

    # Extract article title and text from URL
    article_title, article_text = extract_article_text(url)

    # Clean text using stop words list
    cleaned_text = clean_text(article_text)

    # Analyze text and compute variables
    analysis_result = analyze_text(cleaned_text)

    # (
    #     positive_score,
    #     negative_score,
    #     polarity_score,
    #     subjectivity_score,
    #     avg_sentence_length,
    #     percentage_complex_words,
    #     fog_index,
    #     avg_words_per_sentence,
    #     complex_word_count,
    #     word_count,
    #     syllable_per_word,
    #     personal_pronouns,
    #     avg_word_length,
    # ) = analyze_text(article_text)

    # Append results to output DataFrame
    result_row = {"URL_ID": url_id, "URL": url}
    result_row.update(
        {col: val for col, val in zip(output_columns[2:], analysis_result)}
    )
    output_df = output_df.append(result_row, ignore_index=True)

    # Save cleaned text to a file
    with open(f"{url_id}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(article_title + "\n\n")
        text_file.write(cleaned_text)

# Save output DataFrame to Excel file
output_df.to_excel("Output.xlsx", index=False)