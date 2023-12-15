import config

import tiktoken
import openai
import asyncio

openai.api_key = config.openai_api_key

CHAT_MODES = config.chat_modes

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.82,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0.1,
    "presence_penalty": 0
}

class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="herta"):

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._gen_chat_msg(message, dialog_messages, chat_mode)
                r = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    **OPENAI_COMPLETION_OPTIONS
                )
                answer = r.choices[0].message["content"]

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="herta"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._gen_msg_herta(message, dialog_messages, chat_mode)
                r_gen = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **OPENAI_COMPLETION_OPTIONS
                )

                answer = ""
                async for r_item in r_gen:
                    delta = r_item.choices[0].delta
                    if "content" in delta:
                        answer += delta.content
                        n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                        yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)
                
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer
    

    def _gen_msg_herta(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        if len(dialog_messages)>=2:
            for dialog_message in dialog_messages[config.message_memory*(-1):]: 
                messages.append({"role": "user", "content": dialog_message["user"]})
                messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages
    
    def _gen_chat_msg(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        if "gpt-3.5-turbo" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.encoding_for_model(model)

        if "gpt-3.5-turbo" in model:
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]


async def generate_images(prompt, n_images=4):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size="512x512")
    image_urls = [item.url for item in r.data]
    return image_urls

async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())

async def get_embedding(text, max_retries=5, model="text-embedding-ada-002"):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await openai.Embedding.acreate(input=text, model=model)
            return response.data[0]["embedding"]
        except openai.error.RateLimitError as e:
            retry_after = e.headers.get("Retry-After", 10)
            print(f"Rate limit exceeded; sleeping for {retry_after} seconds and trying again... {e}")
            await asyncio.sleep(retry_after)
            retry_count += 1
    print(f"Failed to process text: {text}. Maximum retry attempts exceeded.")
    return None

async def embed_multiple(inputs):
    tasks = [get_embedding(text) for text in inputs]
    embeddings = await asyncio.gather(*tasks)
    print("Embedding completed.")
    return embeddings