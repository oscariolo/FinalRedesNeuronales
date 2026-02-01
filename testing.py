#Este archivo solo sirve para verificar las predicciones individualmente porsiacaso
#el pipeline esta haciendo huevadas

from transformers import pipeline

# Load the classification pipeline with the specified model
pipe = pipeline("text-classification", model="DeepLearning/fine_tuned_models/Roberta_finetuned_20260201_181547")

# Classify a new sentence
sentence = "Este es un gran día para aprender sobre redes neuronales."
result = pipe(sentence)

# Print the result
print(result)
