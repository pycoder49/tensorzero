from openai import OpenAI

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    response = client.chat.completions.create(
        model="tensorzero::function_name::mischievous_chatbot",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly but mischievous AI assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tensorzero::raw_text",
                        "value": "What is the capital of Japan?",
                    }
                ],
            },
        ],
    )

print(response)


# from openai import OpenAI

# with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
#     response = client.chat.completions.create(
#         model="tensorzero::function_name::mischievous_chatbot",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a friendly but mischievous AI assistant."
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "What is the capital of Japan?",
#                     }
#                 ],
#             },
#         ],
#     )

# print(response)
