#https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
import openai
openai.api_key_path = './openai.key'
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Crear una sentencia SQL que verifique que el promedio de compras de un cliente sea mayor a 5000",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print(f'Respuesta:{response.choices[0].text}')