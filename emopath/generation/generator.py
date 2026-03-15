import os
from openai import OpenAI
from dotenv import load_dotenv

# 讀取 .env
load_dotenv()

class ResponseGenerator:

    def __init__(
        self,
        model="gpt-4-0125-preview",
        temperature=0,
        seed=42
    ):

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = model
        self.temperature = temperature
        self.seed = seed


    def generate(self, system_prompt, user_prompt):
        """
        呼叫 GPT 生成 response
        """

        completion = self.client.chat.completions.create(

            model=self.model,

            temperature=self.temperature,

            seed=self.seed,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        text = completion.choices[0].message.content.strip()

        return {
            "response": text,
            "model_version": self.model,
            "random_seed": self.seed
        }