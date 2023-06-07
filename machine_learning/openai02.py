#https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
import openai
openai.api_key_path = './openai.key'
response = openai.Completion.create(
  model="text-davinci-003",
  #prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:",
  prompt="Clasificar el sentimiento del siguiente Tweet's \n\nTweet: \"el club guarani no será eliminado\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)
print(f'Respuesta:{response}')