# %%
from spacy import displacy
import bs4
import requests
import pandas as pd
import spacy


text = "She couldn't decide if the glass was half empty or half full so she just drank it."



# %%

# Load the en_core_web_sm model
nlp = spacy.load("en_core_web_sm")


# Process the text
doc = nlp(text)


# Print tokens
for token in doc:
    print(token.text)

# %%
# Use beautifulsoup 
# Create a BeautifulSoup object from the HTML
webpage = "https://en.wikipedia.org/wiki/McLaren"
html_doc = requests.get(webpage).content



# %%
soup = bs4.BeautifulSoup(html_doc, 'html.parser')

# Get the text out of the soup and print it
text = soup.find_all("p")[1].text


# Named Entity Recognition
# Process the text with spaCy
doc = nlp(text)


# %%
# Fetch all names entities and their labels
for entity in doc.ents:
    if entity.label_ == "PERSON":
        print(entity.text)



# %%

displacy.render(doc, style="ent", jupyter=True)

df = pd.read_csv("datasets/exp1.csv", encoding="latin-1", header=None)


