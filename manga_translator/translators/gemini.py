import asyncio
import logging
import time
import google.api_core.exceptions
import re
from typing import List

from manga_translator.config import TranslatorConfig

from .config_gpt import ConfigGPT
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import GOOGLE_API_KEY, GOOGLE_HTTP_PROXY

import google.generativeai as genai

logger = logging.getLogger("manga_translator")


class GeminiTranslator(ConfigGPT, CommonTranslator):
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 3
    _MAX_TOKENS = 8192

    # Added models list
    _AVAILABLE_MODELS = [
        "gemini-2.0-pro-exp-0205",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]

    def round_robin_gemini_api_key(self, check_google_key):
        if not GOOGLE_API_KEY and check_google_key:
            raise MissingAPIKeyException("GOOGLE_API_KEY environment variable required")
        self.api_key_index = 0
        self.api_keys = GOOGLE_API_KEY.split(",")
        self.current_api_key = self.api_keys[self.api_key_index]
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)

        return self.current_api_key

    def _configure_safety_settings(self):
        """Configure safety settings to have the lowest possible blocking levels."""
        safety_settings = []
        for category in genai.types.HarmCategory:
            safety_settings.append(
                {
                    "category": category,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
        return safety_settings

    def _configure_generation_config(self):
        """Configure generation settings for the model."""
        return genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self._MAX_TOKENS,
        )

    def get_model(self, to_lang: str, check_google_key=True):
        api_key = self.round_robin_gemini_api_key(check_google_key)
        genai.configure(api_key=api_key)
        self.generation_config = self._configure_generation_config()
        self.safety_settings = self._configure_safety_settings()
        self.logger.info(
            f"Creating model with name {self.model_name} and to_lang {to_lang}, api_key {api_key}, generation_config {self.generation_config}, safety_settings {self.safety_settings}"
        )
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self.chat_system_template.format(to_lang=to_lang),
        )

    def __init__(
        self, model_name: str = "gemini-2.0-flash-exp", check_google_key=True
    ):  # default model is gemini-2.0-flash-exp
        _CONFIG_KEY = "gemini." + model_name
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)

        if model_name not in self._AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {self._AVAILABLE_MODELS}"
            )

        api_key = self.round_robin_gemini_api_key(check_google_key)
        genai.configure(api_key=api_key)
        self.generation_config = self._configure_generation_config()
        self.safety_settings = self._configure_safety_settings()
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
        if GOOGLE_HTTP_PROXY:
            # Configure proxy for requests
            import httpx

            httpx.Client(proxies=f"http://{GOOGLE_HTTP_PROXY}")

        self.token_count = 0
        self.token_count_last = 0
        self._last_request_ts = 0
        self.model_name = model_name  # add model name at instance

    def parse_args(self, args: TranslatorConfig):
        self.config = args.gemini_config

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ""
        if self.include_template:
            prompt = self.prompt_template.format(to_lang=to_lang)
        for i, query in enumerate(queries):
            prompt += f"\n<|{i + 1}|>{query}"
        return [prompt.lstrip()], len(queries)

    async def _translate(
        self, from_lang: str, to_lang: str, queries: List[str]
    ) -> List[str]:
        translations = [""] * len(queries)
        prompt, _ = self._assemble_prompts(from_lang, to_lang, queries)

        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                response_text = await self._request_translation(to_lang, prompt[0])
                translations = self._parse_response(response_text, queries)
                return translations
            except Exception as e:
                self.logger.warning(
                    f"Translation attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self._RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(1)
        return translations

    def _parse_response(self, response: str, queries: List[str]) -> List[str]:
        translations = queries.copy()

        expected_count = len(translations)
        response = self.extract_capture_groups(response, rf"{self.rgx_capture}")
        translation_matches = list(
            re.finditer(r"<\|(\d+)\|>(.*?)(?=(<\|\d+\|>|$))", response, re.DOTALL)
        )

        for match in translation_matches:
            id_num = int(match.group(1))
            translation = match.group(2).strip()
            if id_num < 1 or id_num > expected_count:
                raise ValueError(f"ID {id_num} out of range (1 to {expected_count})")
            translations[id_num - 1] = translation

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        # Build system instruction

        # Build messages using dictionaries instead of Content objects
        messages = []
        if to_lang in self.chat_sample:
            messages.append(
                {"role": "user", "parts": [{"text": self.chat_sample[to_lang][0]}]}
            )
            messages.append(
                {"role": "model", "parts": [{"text": self.chat_sample[to_lang][1]}]}
            )
        messages.append({"role": "user", "parts": [{"text": prompt}]})

        try:
            model = self.get_model(to_lang)
            logger.info(
                {
                    "contents": messages,
                    "generation_config": self.generation_config,
                    "safety_settings": self.safety_settings,
                    "stream": False,
                }
            )
            response = model.generate_content(
                contents=messages,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False,
            )
            response_text = response.text
            logger.debug(
                "\n-- Gemini Response --\n" + response_text + "\n------------------\n"
            )

            if not response.candidates:
                raise ValueError("Empty response from Gemini API")
            return response_text

        except genai.types.BlockedPromptException as e:
            logger.error(f"Blocked prompt error: {e}")
            raise
        except google.api_core.exceptions.RetryError as e:
            self.logger.error(f"Gemini rate limit exceeded or API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60 / self._MAX_REQUESTS_PER_MINUTE
            if now - self._last_request_ts < delay:
                await asyncio.sleep(delay - (now - self._last_request_ts))
            self._last_request_ts = time.time()

    @property
    def available_models(self):
        return self._AVAILABLE_MODELS
