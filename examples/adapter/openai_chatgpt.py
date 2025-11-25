from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()

answer = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': 'milk is good for your health'
        },
        {
            'role': 'assistant',
            'content': 'yes'
        },
        {
            'role': 'user',
            'content': 'milk is bad for your health'
        },
        {
            'role': 'assistant',
            'content': 'yes'
        },
        {
            'role': 'user',
            'content': 'list more reasons'
        }
      ],
    )

print(answer)

answer = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': 'milk is both good and bad for your health'
        },
        {
            'role': 'assistant',
            'content': 'yes'
        },
        {
            'role': 'user',
            'content': 'list more reasons'
        },
      ],
    )
print(answer)