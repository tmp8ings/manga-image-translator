import logging
from typing import Optional

import py3langid as langid

from .config_gpt import ConfigGPT

from .common import *
from .baidu import BaiduTranslator
from .deepseek import DeepseekTranslator

# from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .caiyun import CaiyunTranslator
from .chatgpt import OpenAITranslator
from .nllb import NLLBTranslator, NLLBBigTranslator
from .sugoi import JparacrawlTranslator, JparacrawlBigTranslator, SugoiTranslator
from .m2m100 import M2M100Translator, M2M100BigTranslator
from .mbart50 import MBart50Translator
from .selective import (
    SelectiveOfflineTranslator,
    prepare as prepare_selective_translator,
)
from .none import NoneTranslator
from .original import OriginalTranslator
from .sakura import SakuraTranslator
from .qwen2 import Qwen2Translator, Qwen2BigTranslator
from .groq import GroqTranslator
from .custom_openai import CustomOpenAiTranslator
from .gemini import (
    Gemini2FlashExp,
    Gemini2FlashThinkingExp,
    Gemini2ProExp,
    GeminiTranslator,
)
from ..config import Translator, TranslatorConfig, TranslatorChain
from ..utils import Context

OFFLINE_TRANSLATORS = {
    Translator.offline: SelectiveOfflineTranslator,
    Translator.nllb: NLLBTranslator,
    Translator.nllb_big: NLLBBigTranslator,
    Translator.sugoi: SugoiTranslator,
    Translator.jparacrawl: JparacrawlTranslator,
    Translator.jparacrawl_big: JparacrawlBigTranslator,
    Translator.m2m100: M2M100Translator,
    Translator.m2m100_big: M2M100BigTranslator,
    Translator.mbart50: MBart50Translator,
    Translator.qwen2: Qwen2Translator,
    Translator.qwen2_big: Qwen2BigTranslator,
}

TRANSLATORS = {
    # 'google': GoogleTranslator,
    Translator.youdao: YoudaoTranslator,
    Translator.baidu: BaiduTranslator,
    Translator.deepl: DeeplTranslator,
    Translator.papago: PapagoTranslator,
    Translator.caiyun: CaiyunTranslator,
    Translator.chatgpt: OpenAITranslator,
    Translator.none: NoneTranslator,
    Translator.original: OriginalTranslator,
    Translator.sakura: SakuraTranslator,
    Translator.deepseek: DeepseekTranslator,
    Translator.groq: GroqTranslator,
    Translator.custom_openai: CustomOpenAiTranslator,
    Translator.gemini_pro: Gemini2ProExp,
    Translator.gemini_flash: Gemini2FlashExp,
    Translator.gemini_thinking: Gemini2FlashThinkingExp,
    **OFFLINE_TRANSLATORS,
}

translator_cache = {}

logger = logging.getLogger("manga_translator")


def get_translator(key: Translator, *args, **kwargs) -> CommonTranslator:
    # logger.info(key)
    # logger.info(TRANSLATORS)
    if key not in TRANSLATORS:
        raise ValueError(
            f'Could not find translator for: "{key}". Choose from the following: %s'
            % ",".join(TRANSLATORS)
        )
    if key not in translator_cache:
        translator_factory = TRANSLATORS[key]
        translator_cache[key] = translator_factory(*args, **kwargs)
    return translator_cache[key]


prepare_selective_translator(get_translator)


async def prepare(chain: TranslatorChain):
    # logger.info(chain)
    # logger.info(chain.chain)
    for key, tgt_lang in chain.chain:
        translator = get_translator(key)
        translator.supports_languages("auto", tgt_lang, fatal=True)
        if isinstance(translator, OfflineTranslator):
            await translator.download()


def reset_translator_samples(
    translator: OfflineTranslator | CommonTranslator
):
    logger.debug(
        f"Resetting samples for {translator.__class__.__name__} is configGpt: {isinstance(translator, ConfigGPT)}"
    )

    if isinstance(translator, ConfigGPT):
        try:
            translator.refresh_chat_sample()
        except Exception as e:
            logger.error(
                f"Error resetting samples for {translator.__class__.__name__}: {e}"
            )


async def reset_samples(chain: TranslatorChain):
    logger.debug(f"Resetting samples...")
    try:
        translators = [get_translator(key) for key in chain.chain]
        for translator in translators:
            reset_translator_samples(translator)
    except Exception as e:
        logger.error(f"Error resetting samples: {str(e)}", exc_info=True)


# TODO: Optionally take in strings instead of TranslatorChain for simplicity
async def dispatch(
    chain: TranslatorChain,
    queries: List[str],
    translator_config: Optional[TranslatorConfig] = None,
    use_mtpe: bool = False,
    args: Optional[Context] = None,
    device: str = "cpu",
) -> List[str]:
    if not queries:
        return queries

    if chain.target_lang is not None:
        text_lang = ISO_639_1_TO_VALID_LANGUAGES.get(
            langid.classify("\n".join(queries))[0]
        )
        translator = None
        for key, lang in chain.chain:
            if text_lang == lang:
                translator = get_translator(key)
                break
        if translator is None:
            translator = get_translator(chain.langs[0])
        if isinstance(translator, OfflineTranslator):
            await translator.load("auto", chain.target_lang, device)
        if translator_config:
            translator.parse_args(translator_config)
        queries = await translator.translate(
            "auto", chain.target_lang, queries, use_mtpe
        )
        return queries
    if args is not None:
        args["translations"] = {}
    for key, tgt_lang in chain.chain:
        translator = get_translator(key)
        if isinstance(translator, OfflineTranslator):
            await translator.load("auto", tgt_lang, device)
        if translator_config:
            translator.parse_args(translator_config)
        queries = await translator.translate("auto", tgt_lang, queries, use_mtpe)
        if args is not None:
            args["translations"][tgt_lang] = queries
    return queries


LANGDETECT_MAP = {
    "zh-cn": "CHS",
    "zh-tw": "CHT",
    "cs": "CSY",
    "nl": "NLD",
    "en": "ENG",
    "fr": "FRA",
    "de": "DEU",
    "hu": "HUN",
    "it": "ITA",
    "ja": "JPN",
    "ko": "KOR",
    "pl": "PLK",
    "pt": "PTB",
    "ro": "ROM",
    "ru": "RUS",
    "es": "ESP",
    "tr": "TRK",
    "uk": "UKR",
    "vi": "VIN",
    "ar": "ARA",
    "hr": "HRV",
    "th": "THA",
    "id": "IND",
    "tl": "FIL",
}


async def unload(key: Translator):
    translator_cache.pop(key, None)
