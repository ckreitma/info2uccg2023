#https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
import openai
openai.api_key_path = './openai.key'
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Crear una red neuronal en PyTorch para leer un archivo de 100 entradas con dos capas intermedias y 5 salidas",
  temperature=0.3,
  max_tokens=3000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print(f'Respuesta:{response.choices[0].text}')