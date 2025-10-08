import numpy as np
import asyncio
from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY

class OpenAIEmbedder:
    def __init__(self, api_key=None, model_name="text-embedding-3-small", batch_size=64):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = model_name
        self.batch_size = batch_size


    async def aembed(self, texts):
        async def embed_batch(batch):
            resp = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            return [item.embedding for item in resp.data]

        # Split texts into batches
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])
        vectors = [vec for batch in results for vec in batch]
        return np.array(vectors, dtype="float32")


    def embed(self, texts):
        return asyncio.run(self.aembed(texts))
