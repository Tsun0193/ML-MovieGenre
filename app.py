#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gradio as gr
import warnings
import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification, AutoModelForSeq2SeqLM
from tqdm import tqdm
from torchvision import models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing import image
from torchmetrics.classification import MultilabelF1Score
from sklearn.metrics import average_precision_score, ndcg_score


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


genres = ["Crime", "Thriller", "Fantasy", "Horror", "Sci-Fi", "Comedy", "Documentary", "Adventure", "Film-Noir", "Animation", "Romance", "Drama", "Western", "Musical", "Action", "Mystery", "War", "Children\'s"]
mapping = {}
for i in range(len(genres)):
    mapping[i] = genres[i]
mapping


# In[4]:


tokenizer_gen = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model_gen = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

tokenizer1 = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model1 = DistilBertForSequenceClassification .from_pretrained("distilbert-base-uncased", problem_type="multi_label_classification", num_labels=18)
model1.config.id2label = mapping

tokenizer2 = AutoTokenizer.from_pretrained("dduy193/plot-classification")
model2 = AutoModelForSequenceClassification.from_pretrained("dduy193/plot-classification")
model2.config.id2label = mapping

model3 = models.resnet101(pretrained=False)
model3.fc = torch.nn.Linear(2048, len(genres))


# In[5]:


class Multimodal(torch.nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.fc1 = torch.nn.Linear(18, 18)
        self.fc2 = torch.nn.Linear(18, 18)
        self.fc3 = torch.nn.Linear(18, 18)

    def forward(self, 
                title_input_ids, title_attention_mask,
                plot_input_ids, plot_attention_mask,
                image_input):
        title_output = self.model1(title_input_ids, title_attention_mask)
        plot_output = self.model2(plot_input_ids, plot_attention_mask)
        image_output = self.model3(image_input)

        title_output = self.fc1(title_output.logits)
        plot_output = self.fc2(plot_output.logits)
        image_output = self.fc3(image_output)
        
        output = torch.add(title_output, plot_output)
        output = torch.add(output, image_output)
        return output

# **_PLEASE INSTALL THE MODEL CHECKPOINT FROM THE LINK IN README.txt_**

# In[7]:

model = Multimodal(model1, model2, model3)
model.load_state_dict(torch.load('multimodel.pt', map_location=torch.device('cpu')))
model.eval()
device = torch.device('cpu')


# In[8]:


def generate_plot(title: str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, device) -> str:
    quote = 'What is the story of the movie {}?'
    model_gen.to(device)
    model_gen.eval()

    input_ids = tokenizer(quote.format(title), return_tensors='pt').input_ids.to(device)
    output = model.generate(input_ids, max_length=256, do_sample=True, temperature=0.09)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# In[9]:


def inference(title, image, 
              tokenizer1=tokenizer1, tokenizer2=tokenizer2, tokenizer_gen=tokenizer_gen,
              model_gen=model_gen, model=model, 
              genres=genres, device=device):
    title_input = tokenizer1(title, return_tensors='pt', padding=True, truncation=True)
    title_input_ids = title_input['input_ids'].to(device)
    title_attention_mask = title_input['attention_mask'].to(device)

    plot = generate_plot(title, model_gen, tokenizer_gen, device)
    plot_input = tokenizer2(plot, return_tensors='pt', padding=True, truncation=True)
    plot_input_ids = plot_input['input_ids'].to(device)
    plot_attention_mask = plot_input['attention_mask'].to(device)

    # If image is not uploaded
    if image is None:
        image_input = torch.zeros((1, 3, 224, 224)).to(device)

    else:
        image_input = image.resize((224, 224))
        image_input = v2.ToTensor()(image_input)
        image_input = image_input.unsqueeze(0)
        image_input = image_input.to(device)

    output = model(title_input_ids, title_attention_mask, plot_input_ids, plot_attention_mask, image_input)
    output = torch.sigmoid(output)
    output = output.cpu().detach().numpy()
    output = np.where(output > 0.5, 1, 0)
    output = output.squeeze()
    output = np.where(output == 1)[0]
    output = [genres[i] for i in output]
    return output


# In[10]:


app = gr.Interface(fn=inference, inputs=["text", "pil"], outputs="text", title="Movie Genre Classification", 
                   description="This model classifies the genre of a movie based on its title and poster.", 
                   examples=[["The Matrix", "https://upload.wikimedia.org/wikipedia/en/c/c1/The_Matrix_Poster.jpg"],
                             ["The Dark Knight", "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg"],
                             ["The Godfather", "https://upload.wikimedia.org/wikipedia/en/1/1c/Godfather_ver1.jpg"],
                             ["The Shawshank Redemption", "https://upload.wikimedia.org/wikipedia/en/8/81/ShawshankRedemptionMoviePoster.jpg"],
                             ["The Lord of the Rings: The Return of the King", "https://upload.wikimedia.org/wikipedia/en/2/23/The_Lord_of_the_Rings%2C_TROTK_%282003%29.jpg"],
                             ["The Godfather: Part II", "https://upload.wikimedia.org/wikipedia/en/0/03/Godfather_part_ii.jpg"]])


# In[11]:


app.launch(share=True)

