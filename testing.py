#Este archivo solo sirve para verificar las predicciones individualmente porsiacaso
#el pipeline esta haciendo huevadas

from transformers import pipeline

# Load the classification pipeline with the specified model
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

# Classify a new sentence
sentence = "Esta del putas tu pelo"
result = pipe(sentence)

# Print the result
print(result)
