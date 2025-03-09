import re
from typing import List
from .log import get_logger
from .textblock import TextBlock
from janome.tokenizer import Tokenizer


logger = get_logger(__name__)


tokenizer = Tokenizer()


def is_japanese_onomatopoeia(text: str) -> bool:
    """
    일본어 만화 효과음(의성어/의태어)인지 판단합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        bool: 효과음일 경우 True, 아닐 경우 False
    """
    # 텍스트 정제
    text = text.strip()
    if not text:
        return False

    common_onomatopoeia = [
        "ドキ",
        "バキ",
        "ガシ",
        "ザワ",
        "ゴゴ",
        "ドド",
        "シュ",
        "ガタ",
        "バタ",
        "ジー",
        "カチ",
        "ピキ",
        "バサ",
        "パタ",
        "ズズ",
        "ブブ",
        "キラ",
        "パキ",
        "ガタン",
        "ドカン",
        "ボン",
        "キュ",
        "シュン",
        "バン",
        "ガン",
        "ドン",
        "ギリ",
        "ゴト",
        "ピシ",
        "フワ",
        "ポン",
        "ハア",
        "パア",
        "ザッ",
        "ふリン",
    ]

    for sfx in common_onomatopoeia:
        if text == sfx:
            return True

    if len(text) >= 5:
        return False

    # 6. 일반적인 효과음 패턴
    common_patterns = [
        r"[ァ-ヺ]+ッ$",
        r"[ぁ-ゖ]+っ$",
        r"[ァ-ヺ]+ー[ァ-ヺ]",
        r"[ァ-ヺ]+ー$",
    ]

    for pattern in common_patterns:
        if re.search(pattern, text):
            return True

    return False


def is_onomatopoeia(text: str):
    result = is_japanese_onomatopoeia(text)
    logger.debug(f"Text: {text}, Is Onomatopoeia: {result}")
    return result


def filter_onomatopoeia(text_blocks: List[TextBlock]):
    return [tb for tb in text_blocks if not is_onomatopoeia(tb.text)]
