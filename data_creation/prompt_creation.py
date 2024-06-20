from openai import OpenAI

# client = OpenAI(
#     api_key = "sk-dM6Ho34RUEMQW8Q2wIUTEFxbZqeOoArDDBVWzFmfeHsMrRId",
#     base_url = "https://api.fe8.cn/v1"
# )

client = OpenAI(
    api_key = "sk-yAA9L6BaNLhI8UqZFf1a509f9bF94b9bBf85Ff1826D6E4Bc",
    base_url = "http://rerverseapi.workergpt.cn/v1",
    timeout=1000
)

source_prompt = 'A group of marmoset monkeys are sitting on a branch in an enclosure.'
edit = 'Make it in the Glassmorphism style.'

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "This is a caption of a video:" + source_prompt + "." + 
                "If I want to edit the video according to this instruction :" + edit + "." +
                "please provide a new caption of the edited video.",
        }
    ],
    model="gpt-4-1106-preview",
    # model="gpt-3.5-turbo",
)
print(chat_completion.choices[0].message.content)
