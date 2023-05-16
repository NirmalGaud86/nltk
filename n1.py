#!/usr/bin/env python
# coding: utf-8

# In[9]:


import gradio as gr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data (if not already downloaded)
nltk.download('vader_lexicon')

# Create the model
analyzer = SentimentIntensityAnalyzer()

# Define the prediction function
def predict_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

# Create a Gradio interface
input_text = gr.inputs.Textbox(label="Enter text")
output_text = gr.outputs.Textbox(label="Sentiment")
interface = gr.Interface(fn=predict_sentiment, inputs=input_text, outputs=output_text, title="Sentiment Analysis")

# Launch the Gradio interface
interface.launch()


# In[ ]:




