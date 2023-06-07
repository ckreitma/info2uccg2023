#https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
import openai
openai.api_key_path = './openai.key'
messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {chat}")
    messages.append({"role": "assistant", "content": reply})