from dotenv import dotenv_values

conf = dotenv_values()

OPENAI_API_KEY = conf.get('OPENAI_API_KEY', '')