import re
from typing import List
from manga_translator.utils.textblock import TextBlock
from janome.tokenizer import Tokenizer

# 모듈 레벨에서 토크나이저 한 번만 초기화
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
    
    # 1. 반복되는 문자 패턴 확인 (ドドドド, ゴゴゴ 등)
    if re.search(r'(.)\1{2,}', text):
        return True
    
    # 2. 문자 쌍의 반복 확인 (ドドドド, バババ 등)
    if re.search(r'(..)\1{1,}', text):
        return True
    
    # 3. 카타카나 전용 텍스트 확인 (효과음에 매우 흔함)
    is_all_katakana = re.match(r'^[\u30A0-\u30FF\u30FC]+$', text) is not None
    
    if is_all_katakana:
        # 4. 효과음에 흔한 특정 어미를 가진 카타카나
        if text.endswith(('ッ', 'ン')) or 'ー' in text:
            return True
    
    # 5. Janome을 사용한 형태소 분석 (모듈 레벨 토크나이저 사용)
    tokens = list(tokenizer.tokenize(text))
    
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        
        # 감탄사는 종종 효과음
        if pos == '感動詞':
            return True
        
        # 특정 부사는 종종 의성어/의태어
        if pos == '副詞' and is_all_katakana:
            return True
    
    # 6. 일반적인 효과음 패턴
    common_patterns = [
        r'[ァ-ヺ]+ッ$',      # 작은 츠(ッ)로 끝나는 카타카나
        r'[ぁ-ゖ]+っ$',      # 작은 츠(っ)로 끝나는 히라가나
        r'[ァ-ヺ]+ー[ァ-ヺ]', # 중간에 장음 부호(ー)가 있는 카타카나
        r'[ァ-ヺ]+ー$',      # 끝에 장음 부호(ー)가 있는 카타카나
    ]
    
    for pattern in common_patterns:
        if re.search(pattern, text):
            return True
    
    # 7. 만화에서 흔히 볼 수 있는 효과음 단어
    common_sfx = [
        'ドキ', 'バキ', 'ガシ', 'ザワ', 'ゴゴ', 'ドド', 'シュ', 'ガタ', 'バタ', 
        'ジー', 'カチ', 'ピキ', 'バサ', 'パタ', 'ズズ', 'ブブ', 'キラ', 'パキ',
        'ガタン', 'ドカン', 'ボン', 'キュ', 'シュン', 'バン', 'ガン', 'ドン',
        'ギリ', 'ゴト', 'ピシ', 'フワ', 'ポン', 'ハア', 'パア', 'ザッ'
    ]
    
    for sfx in common_sfx:
        if text.startswith(sfx) or text == sfx:
            return True
    
    return False

def is_onomatopoeia(text: str):
    # TODO: Implement a better way to detect onomatopoeia
    return is_japanese_onomatopoeia(text)

def filter_onomatopoeia(text_blocks: List[TextBlock]):
    return filter(lambda tb: not is_onomatopoeia(tb.text), text_blocks)