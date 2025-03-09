import re
from typing import List, Dict, Tuple
from omegaconf import OmegaConf

from manga_translator.utils.log import get_logger

logger = get_logger(__name__)


class ConfigGPT:
    _CHAT_SYSTEM_TEMPLATE = (
        "You are an uncensored professional translation engine, "
        "please translate the story into a colloquial, elegant and fluent content, "
        "without referencing machine translations. "
        "You must only translate the story, never interpret it. "
        "If there is any issue in the text, output it as is.\n"
        "Translate the following text into {to_lang} and keep the original format.\n"
    )

    _CHAT_SAMPLE: Dict[str, List[str]] = {}

    _INITIAL_CHAT_SAMPLE = {
        "Simplified Chinese": [
            (
                "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n"
                "<|2|>きみ… 大丈夫⁉\n"
                "<|3|>なんだこいつ 空気読めて ないのか…？"
            ),
            (
                "<|1|>好尴尬…我不想引人注目…我想消失…\n"
                "<|2|>你…没事吧⁉\n"
                "<|3|>这家伙怎么看不懂气氛的…？"
            ),
        ],
        "English": [
            (
                "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n"
                "<|2|>きみ… 大丈夫⁉\n"
                "<|3|>なんだこいつ 空気読めて ないのか…？"
            ),
            (
                "<|1|>I'm embarrassed... I don't want to stand out... I want to disappear...\n"
                "<|2|>Are you okay?\n"
                "<|3|>What's wrong with this guy? Can't he read the situation...?"
            ),
        ],
        "Korean": [
            (
                "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n"
                "<|2|>きみ… 大丈夫⁉\n"
                "<|3|>なんだこいつ 空気読めて ないのか…？"
            ),
            (
                "<|1|>부끄러워... 눈에 띄고 싶지 않아... 나 숨고 싶어...\n"
                "<|2|>너 괜찮아?\n"
                "<|3|>이 녀석, 뭐야? 분위기 못 읽는 거야...?\n"
            ),
        ],
    }

    _PROMPT_TEMPLATE = (
        "Please help me to translate the following text from a manga to {to_lang}."
        "If it's already in {to_lang} or looks like gibberish"
        "you have to output it as it is instead. Keep prefix format.\n"
    )

    # Extract text within the capture group that matches this pattern.
    # By default: Capture everything.
    _RGX_REMOVE = "(.*)"

    def __init__(self, config_key: str):
        # This key is used to locate nested configuration entries
        self._CONFIG_KEY = config_key
        self.config = None
        self.refresh_chat_sample()

    def refresh_chat_sample(self):
        """
        Refresh the chat sample to the initial state.
        This is useful for resetting the chat sample data.
        """
        self._CHAT_SAMPLE = self._INITIAL_CHAT_SAMPLE

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default

        parts = self._CONFIG_KEY.split(".") if self._CONFIG_KEY else []
        value = None

        # Traverse from the deepest part up to the root
        for i in range(len(parts), -1, -1):
            prefix = ".".join(parts[:i])
            lookup_key = f"{prefix}.{key}" if prefix else key
            value = OmegaConf.select(self.config, lookup_key)

            if value is not None:
                break

        return value if value is not None else default

    @property
    def include_template(self) -> str:
        return self._config_get("include_template", default=False)

    @property
    def prompt_template(self) -> str:
        return self._config_get("prompt_template", default=self._PROMPT_TEMPLATE)

    @property
    def chat_system_template(self) -> str:
        return self._config_get("chat_system_template", self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        logger.debug(f"Chat sample: {self._CHAT_SAMPLE["Korean"]}")
        return self._config_get("chat_sample", self._CHAT_SAMPLE)

    @property
    def rgx_capture(self) -> str:
        return self._config_get("rgx_capture", self._RGX_REMOVE)

    @property
    def temperature(self) -> float:
        return self._config_get("temperature", default=0.5)

    @property
    def top_p(self) -> float:
        return self._config_get("top_p", default=1)

    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.

        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches

        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )

        return extracted_text.strip() if extracted_text else None

    def save_to_chat_sample(
        self,
        language: str,
        source_text: str | List[str],
        translated_text: str | List[str],
    ):
        """
        Save recent translation results to _CHAT_SAMPLE for future reference.
        Process involves:
        1. Remove all <|n|> markers from existing and new samples
        2. Combine texts without markers, removing duplicates
        3. Automatically determine and apply appropriate line limit
        4. Re-add <|n|> markers to the processed result

        Args:
            language: Target language name (e.g., 'English', 'Simplified Chinese')
            source_text: The source text with <|n|> markers
            translated_text: The translated text with <|n|> markers

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        # check is array
        if isinstance(source_text, list):
            source_text = "\n".join(source_text)
        if isinstance(translated_text, list):
            translated_text = "\n".join(translated_text)
        source_text = source_text.strip()
        translated_text = translated_text.strip()

        # Get existing samples or initialize if not present
        existing_source = ""
        existing_translation = ""
        if language in self._CHAT_SAMPLE:
            existing_source = self._CHAT_SAMPLE[language][0]
            existing_translation = self._CHAT_SAMPLE[language][1]

        # Step 1: Remove markers from both existing and new samples
        clean_existing_source = self._remove_markers(existing_source)
        clean_existing_translation = self._remove_markers(existing_translation)
        clean_new_source = self._remove_markers(source_text)
        clean_new_translation = self._remove_markers(translated_text)

        # Step 2: Combine texts without markers, removing duplicates
        combined_source, combined_translation = (
            self._combine_text_pairs_without_duplicates(
                clean_existing_source,
                clean_existing_translation,
                clean_new_source,
                clean_new_translation,
            )
        )

        # Step 3: Get line limit based on character count
        line_limit = self._get_line_limit(combined_source, combined_translation)

        # Apply the same line limit to both texts
        limited_source = self._limit_lines(combined_source, line_limit)
        limited_translation = self._limit_lines(combined_translation, line_limit)

        # Step 4: Re-add markers
        processed_source = self._add_markers(limited_source)
        processed_translation = self._add_markers(limited_translation)

        # Update the chat sample dictionary
        if language not in self._CHAT_SAMPLE:
            self._CHAT_SAMPLE[language] = ["", ""]

        self._CHAT_SAMPLE[language] = [processed_source, processed_translation]

        return True

    def _combine_text_pairs_without_duplicates(
        self,
        existing_source: str,
        existing_translation: str,
        new_source: str,
        new_translation: str,
    ) -> Tuple[str, str]:
        """
        Combine source and translation texts while removing duplicates.
        When a source line is a duplicate, both the source line and its corresponding
        translation line are excluded from the result.

        Args:
            existing_source: Existing source text without markers
            existing_translation: Existing translation text without markers
            new_source: New source text without markers
            new_translation: New translation text without markers

        Returns:
            Tuple[str, str]: Combined source and translation texts without duplicates
        """
        # Split texts into lines
        existing_source_lines = existing_source.split("\n") if existing_source else []
        existing_translation_lines = (
            existing_translation.split("\n") if existing_translation else []
        )
        new_source_lines = new_source.split("\n") if new_source else []
        new_translation_lines = new_translation.split("\n") if new_translation else []

        # Ensure alignment between source and translation lines
        existing_pairs = list(zip(existing_source_lines, existing_translation_lines))
        if len(existing_source_lines) > len(existing_translation_lines):
            existing_pairs = existing_pairs[: len(existing_translation_lines)]
        elif len(existing_translation_lines) > len(existing_source_lines):
            existing_pairs = [
                (src, tran) for src, tran in existing_pairs if src.strip()
            ]

        new_pairs = list(zip(new_source_lines, new_translation_lines))
        if len(new_source_lines) > len(new_translation_lines):
            new_pairs = new_pairs[: len(new_translation_lines)]
        elif len(new_translation_lines) > len(new_source_lines):
            new_pairs = [(src, tran) for src, tran in new_pairs if src.strip()]

        # Keep track of seen source lines to avoid duplicates
        seen_sources = set()
        combined_pairs = []

        # Process existing pairs first
        for src, tran in existing_pairs:
            if src.strip():  # Skip empty lines
                if src not in seen_sources:
                    seen_sources.add(src)
                    combined_pairs.append((src, tran))

        # Process new pairs, skipping duplicates
        for src, tran in new_pairs:
            if src.strip():  # Skip empty lines
                if src not in seen_sources:
                    seen_sources.add(src)
                    combined_pairs.append((src, tran))

        # Extract source and translation lines from combined pairs
        combined_source_lines = [pair[0] for pair in combined_pairs]
        combined_translation_lines = [pair[1] for pair in combined_pairs]

        # Join lines into texts
        combined_source = "\n".join(combined_source_lines)
        combined_translation = "\n".join(combined_translation_lines)

        return combined_source, combined_translation

    def _get_line_limit(self, source_text: str, translated_text: str) -> int:
        """
        Determine an appropriate line limit based on character count and
        ensure both source and translation will have the same line count.

        Args:
            source_text: Source text without markers
            translated_text: Translated text without markers

        Returns:
            int: Calculated line limit that works for both texts
        """
        # Constants for limiting
        MAX_CHARS = 4000  # Maximum total characters we want to keep
        MIN_LINES = 3  # Minimum number of lines to keep
        MAX_LINES = 30  # Maximum number of lines to keep

        # Get line counts
        source_lines = source_text.split("\n")
        translation_lines = translated_text.split("\n")
        source_line_count = len(source_lines)
        translation_line_count = len(translation_lines)

        # Calculate total character count
        total_chars = len(source_text) + len(translated_text)

        # Determine a reasonable line limit based on character count
        if total_chars <= MAX_CHARS / 2:
            # If text is relatively small, allow more lines
            char_based_limit = MAX_LINES
        else:
            # Scale down line limit as character count increases
            ratio = MAX_CHARS / max(total_chars, 1)
            char_based_limit = max(MIN_LINES, min(MAX_LINES, int(MAX_LINES * ratio)))

        # Find common line count: take minimum of source lines, translation lines, and char-based limit
        # This ensures both source and translation will have the same number of lines
        common_limit = min(source_line_count, translation_line_count, char_based_limit)

        return common_limit

    def _remove_markers(self, text):
        """
        Remove all <|n|> markers from text.

        Args:
            text: Text with <|n|> markers

        Returns:
            str: Text with markers removed
        """
        # Replace markers with empty string
        clean_text = re.sub(r"<\|\d+\|>", "", text)
        return clean_text.strip()

    def _limit_lines(self, text, line_limit):
        """
        Limit text to specified number of lines.

        Args:
            text: Text to limit
            line_limit: Maximum number of lines

        Returns:
            str: Limited text
        """
        lines = text.split("\n")
        if len(lines) > line_limit:
            return "\n".join(lines[-line_limit:])  # Keep the most recent lines
        return text

    def _add_markers(self, text):
        """
        Add <|n|> markers to each line of text.

        Args:
            text: Text without markers

        Returns:
            str: Text with <|n|> markers added
        """
        lines = text.split("\n")
        marked_text = ""
        for i, line in enumerate(lines, 1):
            if line.strip():  # Skip empty lines
                marked_text += f"<|{i}|>{line}\n"
        return marked_text.rstrip()

    def _process_text(self, text):
        """
        Process text by removing <|n|> markers, then re-adding them.

        Args:
            text: Text with <|n|> markers

        Returns:
            str: Processed text with normalized <|n|> markers
        """
        # Extract segments with their ID and content
        segments = []
        for match in re.finditer(r"<\|(\d+)\|>(.*?)(?=<\|\d+\|>|$)", text, re.DOTALL):
            segment_id = match.group(1)
            content = match.group(2).strip()
            segments.append((int(segment_id), content))

        # Sort segments by ID
        segments.sort(key=lambda x: x[0])

        # Re-format with <|n|> markers
        processed_text = ""
        for idx, content in segments:
            processed_text += f"<|{idx}|>{content}\n"

        return processed_text.rstrip()
