import cv2
from manga_translator.config import Direction
from manga_translator.utils.log import get_logger
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPoint
from functools import cached_property
import copy
import re
import py3langid as langid

from .generic import color_difference, is_right_to_left_char, is_valuable_char

# from ..detection.ctd_utils.utils.imgproc_utils import union_area, xywh2xyxypoly

logger = get_logger("textblock")

# LANG_LIST = ['eng', 'ja', 'unknown']
# LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

# determines render direction
LANGUAGE_ORIENTATION_PRESETS = {
    "CHS": "auto",
    "CHT": "auto",
    "CSY": "h",
    "NLD": "h",
    "ENG": "h",
    "FRA": "h",
    "DEU": "h",
    "HUN": "h",
    "ITA": "h",
    "JPN": "auto",
    "KOR": "auto",
    "PLK": "h",
    "PTB": "h",
    "ROM": "h",
    "RUS": "h",
    "ESP": "h",
    "TRK": "h",
    "UKR": "h",
    "VIN": "h",
    "ARA": "hr",  # horizontal reversed (right to left)
    "FIL": "h",
}


class TextBlock(object):
    """
    Object that stores a block of text made up of textlines.
    """

    def __init__(
        self,
        lines: List[Tuple[int, int, int, int]],
        texts: List[str] = None,
        language: str = "unknown",
        font_size: float = -1,
        angle: float = 0,
        translation: str = "",
        fg_color: Tuple[float] = (0, 0, 0),
        bg_color: Tuple[float] = (0, 0, 0),
        line_spacing=1.0,
        letter_spacing=1.0,
        font_family: str = "",
        bold: bool = False,
        underline: bool = False,
        italic: bool = False,
        direction: str = "auto",
        alignment: str = "auto",
        rich_text: str = "",
        _bounding_rect: List = None,
        default_stroke_width=0.2,
        font_weight=50,
        source_lang: str = "",
        target_lang: str = "",
        opacity: float = 1.0,
        shadow_radius: float = 0.0,
        shadow_strength: float = 1.0,
        shadow_color: Tuple = (0, 0, 0),
        shadow_offset: List = [0, 0],
        prob: float = 1,
        is_rearranged: bool = False,
        **kwargs,
    ) -> None:
        self.lines = np.array(lines, dtype=np.int32)
        # self.lines.sort()
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.texts = texts if texts is not None else []
        self.text = texts[0]
        if self.text and len(texts) > 1:
            for txt in texts[1:]:
                first_cjk = "\u3000" <= self.text[-1] <= "\u9fff"
                second_cjk = txt and ("\u3000" <= txt[0] <= "\u9fff")
                if first_cjk or second_cjk:
                    self.text += txt
                else:
                    self.text += " " + txt
        self.prob = prob

        self.translation = translation

        self.fg_colors = fg_color
        self.bg_colors = bg_color

        # self.stroke_width = stroke_width
        self.font_family: str = font_family
        self.bold: bool = bold
        self.underline: bool = underline
        self.italic: bool = italic
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self._alignment = alignment
        self._source_lang = source_lang
        self.target_lang = target_lang

        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.adjust_bg_color = True

        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset

        self._is_rearranged = is_rearranged

    def __str__(self):
        content = f"TextBlock(text: {self.translation[:3]}, angle: {self.angle}, direction: {self.direction}, alignment: {self.alignment},"
        content = f"{content}\n xyxy: {self.xyxy}, xywh: {self.xywh}, center: {self.center}, aspect_ratio: {self.aspect_ratio},"
        content = f"{content}\n area: {self.area}, real_area: {self.real_area}, polygon_aspect_ratio: {self.polygon_aspect_ratio},"
        content = f"{content})"
        
        content = f"TextBlock(text: {self.translation[:3]}, xyxy: {self.xyxy}, xywh: {self.xywh}, center: {self.center}"
        return content

    def __repr__(self):
        return self.__str__()

    @property
    def xyxy(self):
        """Coordinates of the bounding box"""
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return np.array([x1, y1, x2, y2]).astype(np.int32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.int32)

    @property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    @property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @property
    def unrotated_min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    @property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.clip(0).reshape(-1, 4, 2).astype(np.int64)

    @property
    def polygon_aspect_ratio(self) -> float:
        """width / height"""
        polygons = self.unrotated_polygons.reshape(-1, 4, 2)
        middle_pts = (polygons[:, [1, 2, 3, 0]] + polygons) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        return np.mean(norm_h / norm_v)

    @property
    def unrotated_size(self) -> Tuple[int, int]:
        """Returns width and height of unrotated bbox"""
        middle_pts = (self.min_rect[:, [1, 2, 3, 0]] + self.min_rect) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3])
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0])
        return norm_h, norm_v

    @property
    def aspect_ratio(self) -> float:
        """width / height"""
        return self.unrotated_size[0] / self.unrotated_size[1]

    @property
    def polygon_object(self) -> Polygon:
        min_rect = self.min_rect[0]
        return MultiPoint(
            [
                tuple(min_rect[0]),
                tuple(min_rect[1]),
                tuple(min_rect[2]),
                tuple(min_rect[3]),
            ]
        ).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    @property
    def real_area(self) -> float:
        lines = self.lines.reshape((-1, 2))
        return MultiPoint([tuple(l) for l in lines]).convex_hull.area

    def normalized_width_list(self) -> List[float]:
        polygons = self.unrotated_polygons
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).sum())
        width_list = np.array(width_list)
        width_list = width_list / np.sum(width_list)
        return width_list.tolist()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self):
        blk_dict = copy.deepcopy(vars(self))
        return blk_dict

    def get_transformed_region(
        self, img: np.ndarray, line_idx: int, textheight: int, maxwidth: int = None
    ) -> np.ndarray:
        im_h, im_w = img.shape[:2]

        line = np.round(np.array(self.lines[line_idx])).astype(np.int64)

        x1, y1, x2, y2 = (
            line[:, 0].min(),
            line[:, 1].min(),
            line[:, 0].max(),
            line[:, 1].max(),
        )
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_croped = img[y1:y2, x1:x2]

        direction = "v" if self.src_is_vertical else "h"

        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]  # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]  # horizontal vectors of textlines
        norm_v = np.linalg.norm(vec_v)
        norm_h = np.linalg.norm(vec_h)

        if textheight is None:
            if direction == "h":
                textheight = int(norm_v)
            else:
                textheight = int(norm_h)

        if norm_v <= 0 or norm_h <= 0:
            print("invalid textpolygon to target img")
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == "h":
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(
                np.float32
            )
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print("invalid textpolygon to target img")
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
        elif direction == "v":
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(
                np.float32
            )
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print("invalid textpolygon to target img")
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if maxwidth is not None:
            h, w = region.shape[:2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))

        return region

    @property
    def source_lang(self):
        if not self._source_lang:
            self._source_lang = langid.classify(self.text)[0]
        return self._source_lang

    def get_translation_for_rendering(self):
        text = self.translation
        if self.direction.endswith("r"):
            # The render direction is right to left so left-to-right
            # text/number chunks need to be reversed to look normal.

            text_list = list(text)
            l2r_idx = -1

            def reverse_sublist(l, i1, i2):
                delta = i2 - i1
                for j1 in range(i1, i2 - delta // 2):
                    j2 = i2 - (j1 - i1) - 1
                    l[j1], l[j2] = l[j2], l[j1]

            for i, c in enumerate(text):
                if not is_right_to_left_char(c) and is_valuable_char(c):
                    if l2r_idx < 0:
                        l2r_idx = i
                elif l2r_idx >= 0 and i - l2r_idx > 1:
                    # Reverse left-to-right characters for correct rendering
                    reverse_sublist(text_list, l2r_idx, i)
                    l2r_idx = -1
            if l2r_idx >= 0 and i - l2r_idx > 1:
                reverse_sublist(text_list, l2r_idx, len(text_list))

            text = "".join(text_list)
        return text

    @property
    def is_bulleted_list(self):
        """
        A determining factor of whether we should be sticking to the strict per textline
        text distribution when rendering.
        """
        if len(self.texts) <= 1:
            return False

        bullet_regexes = [
            r"[^\w\s]",  # ○ ... ○ ...
            r"[\d]+\.",  # 1. ... 2. ...
            r"[QA]:",  # Q: ... A: ...
        ]
        bullet_type_idx = -1
        for line_text in self.texts:
            for i, breg in enumerate(bullet_regexes):
                if re.search(r"(?:[\n]|^)((?:" + breg + r")[\s]*)", line_text):
                    if bullet_type_idx >= 0 and bullet_type_idx != i:
                        return False
                    bullet_type_idx = i
        return bullet_type_idx >= 0

    def set_font_colors(self, fg_colors, bg_colors):
        self.fg_colors = np.array(fg_colors)
        self.bg_colors = np.array(bg_colors)

    def update_font_colors(self, fg_colors: np.ndarray, bg_colors: np.ndarray):
        nlines = len(self)
        if nlines > 0:
            self.fg_colors += fg_colors / nlines
            self.bg_colors += bg_colors / nlines

    def get_font_colors(self, bgr=False):

        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)

        if bgr:
            frgb = frgb[::-1]
            brgb = brgb[::-1]

        if self.adjust_bg_color:
            fg_avg = np.mean(frgb)
            if color_difference(frgb, brgb) < 30:
                brgb = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)

        return frgb, brgb

    @property
    def direction(self):
        """Render direction determined through used language or aspect ratio."""
        if self._direction not in ("h", "v", "hr", "vr"):
            d = LANGUAGE_ORIENTATION_PRESETS.get(self.target_lang)
            if d in ("h", "v", "hr", "vr"):
                return d

            if self._direction == Direction.h:
                return "h"
            elif self._direction == Direction.v:
                return "v"

            if self.aspect_ratio < 1:
                return "v"
            else:
                return "h"
        return self._direction

    @property
    def vertical(self):
        return self.direction.startswith("v")

    @property
    def horizontal(self):
        return self.direction.startswith("h")

    @property
    def alignment(self):
        """Render alignment(/gravity) determined through used language."""
        if self._alignment in ("left", "center", "right"):
            return self._alignment
        if len(self.lines) == 1:
            return "center"

        if self.direction == "h":
            return "center"
        elif self.direction == "hr":
            return "right"
        else:
            return "left"

    @property
    def stroke_width(self):
        diff = color_difference(*self.get_font_colors())
        if diff > 15:
            return self.default_stroke_width
        return 0

    def is_vertical_caption(self, img: np.ndarray) -> bool:
        """세로 쓰기 캡션 여부 (aspect ratio, 너비, 배경, 위치 기반으로 판단)"""
        if not (self.aspect_ratio < 0.7):  # 기존 조건 유지
            logger.debug(
                f"Aspect ratio or width not satisfied for {self.translation[:3]}: {self.aspect_ratio}, {self.xywh[2]}"
            )
            return False

        if img is not None:
            x1, y1, x2, y2 = self.xyxy
            region = img[y1:y2, x1:x2]
            if region.size == 0:  # 이미지 영역 벗어난 경우 방지
                logger.debug(f"Region size is 0 for {self.translation[:3]}")
                return False
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            bg_variance = np.var(gray_region)
            if bg_variance < 30:  # 배경 variance 임계값 (조정 가능)
                logger.debug(
                    f"Background variance not satisfied for {self.translation[:3]}: {bg_variance}"
                )
                return False  # 배경 variance가 낮으면 말풍선으로 간주

        # 위치 기반 조건 (상단에서 시작, 세로로 긴 형태)
        image_height = img.shape[0] if img is not None else 500
        if self.xyxy[1] >= image_height * 0.3:
            logger.debug(
                f"Location condition not satisfied for {self.translation[:3]}: {self.xyxy[1]} and image height is {image_height}"
            )
            return False  # 위치 조건 불만족 시 캡션 아님

        logger.debug(f"This is a vertical caption for{self.translation[:3]}: {self.xyxy[1]}, {self.aspect_ratio}")
        return True  # 그 외는 세로 쓰기 캡션임

    @property
    def is_rearranged(self) -> bool:
        """재배치 여부 확인 속성"""
        return self._is_rearranged

    @is_rearranged.setter
    def is_rearranged(self, value: bool):
        """재배치 여부 설정 속성"""
        self._is_rearranged = value


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)

    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated


def sort_regions(regions: List[TextBlock], right_to_left=True) -> List[TextBlock]:
    # Sort regions from right to left, top to bottom
    sorted_regions = []
    for region in sorted(regions, key=lambda region: region.center[1]):
        for i, sorted_region in enumerate(sorted_regions):
            if region.center[1] > sorted_region.xyxy[3]:
                continue
            if region.center[1] < sorted_region.xyxy[1]:
                sorted_regions.insert(i + 1, region)
                break

            # y center of region inside sorted_region so sort by x instead
            if right_to_left and region.center[0] > sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center[0] < sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
        else:
            sorted_regions.append(region)
    return sorted_regions


def find_background_candidates(
    img: np.ndarray, 
    min_width: int = 100, 
    min_height: int = 50,
    quality_threshold: float = 1500
) -> List[Tuple[int, int, int, int]]:
    """
    Find regions with relatively uniform background suitable for text placement.
    
    Args:
        img: Input image
        min_width: Minimum width required for a region
        min_height: Minimum height required for a region
        quality_threshold: Maximum score for a region to be considered viable
        
    Returns:
        List of tuples: (x, y, width, height) representing candidate regions
    """
    h, w = img.shape[:2]
    regions = []
    
    # Define various region sizes to try
    grid_sizes = [(100, 50), (150, 80), (200, 100), (250, 120), (300, 150)]
    step_size = 50
    
    # Convert to grayscale once for efficiency
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate edge map once for efficiency
    edge_map = cv2.Canny(gray_img, 50, 150)
    
    # Scan the image with multiple region sizes
    for grid_w, grid_h in grid_sizes:
        for y in range(0, h - grid_h, step_size):
            for x in range(0, w - grid_w, step_size):
                if x + min_width > w or y + min_height > h:
                    continue
                    
                # Extract region
                region = gray_img[y:y+grid_h, x:x+grid_w]
                region_edges = edge_map[y:y+grid_h, x:x+grid_w]
                
                # Calculate metrics for region quality
                variance = np.var(region)  # Lower variance = more uniform color
                edge_density = np.sum(region_edges > 0) / (grid_w * grid_h)  # Lower = fewer edges
                
                # Combined score (lower is better) - relaxed criteria
                score = variance * 0.5 + edge_density * 800
                
                # Only add regions below the quality threshold
                if score < quality_threshold:
                    regions.append((x, y, grid_w, grid_h))
    
    return regions


def rearrange_vertical_text_to_horizontal(
    text_blocks: List[TextBlock], img: np.ndarray
) -> List[TextBlock]:
    """
    Rearranges vertical text blocks to horizontal orientation,
    maintaining reading order and optimizing placement.
    
    Args:
        text_blocks: List of text blocks
        img: Input image
        
    Returns:
        List of text blocks with vertical captions rearranged horizontally
    """
    img_height, img_width = img.shape[:2]
    
    # Separate vertical captions from other text blocks
    vertical_caption_blocks: List[TextBlock] = []
    horizontal_blocks: List[TextBlock] = []
    
    for block in text_blocks:
        if block.is_vertical_caption(img):
            vertical_caption_blocks.append(block)
        else:
            horizontal_blocks.append(block)
    
    # If no vertical captions to rearrange, return original blocks
    if not vertical_caption_blocks:
        return text_blocks
    
    # Sort vertical blocks by reading order (for manga: right-to-left, top-to-bottom)
    vertical_caption_blocks.sort(key=lambda b: (-b.xyxy[0], b.xyxy[1]))
    
    # Get existing block bounding boxes to avoid overlapping
    existing_bbox_list = [block.xyxy.tolist() for block in horizontal_blocks]
    
    # Calculate adjusted sizes for horizontal layout
    adjusted_block_sizes = []
    
    for block in vertical_caption_blocks:
        # For vertical text becoming horizontal, we often need to swap width and height
        # with some adjustments for readability
        orig_width, orig_height = block.xywh[2], block.xywh[3]
        
        # Text that was tall and narrow will be wide and short in horizontal layout
        # This ratio helps determine how much to adjust dimensions
        aspect_ratio = orig_height / max(orig_width, 1)
        
        # Calculate new dimensions based on content
        # For very tall blocks, make them wider but not too short
        if aspect_ratio > 4:
            # Very tall block becomes moderately wide
            new_width = int(orig_height * 0.7)
            new_height = int(orig_width * 1.5)
        else:
            # Normal adjustment
            new_width = int(orig_height * 0.9)
            new_height = int(orig_width * 1.2)
        
        # Scale text size appropriately
        char_count = len(block.translation.strip())
        if char_count > 0:
            # Ensure width accommodates longer text
            min_width_per_char = 10  # Minimum pixels per character
            text_width_needed = char_count * min_width_per_char
            new_width = max(new_width, text_width_needed)
        
        # Ensure minimum readable size
        new_width = max(new_width, 50)
        new_height = max(new_height, 20)
        
        adjusted_block_sizes.append((new_width, new_height))
    
    # Record original positions for reference
    original_positions = [block.xyxy for block in vertical_caption_blocks]
    original_centers = [block.center for block in vertical_caption_blocks]
    
    # Find background candidate regions
    candidate_regions = find_background_candidates(
        img, 
        min_width=100,
        min_height=max([h for _, h in adjusted_block_sizes]) + 20 if adjusted_block_sizes else 50
    )
    
    if not candidate_regions:
        logger.warning("No background candidates found. Using default regions.")
        # Create default regions if none found
        candidate_regions = [
            (10, 10, img_width - 20, 100),
            (10, img_height - 110, img_width - 20, 100)
        ]
    
    # Try to find best placement within candidate regions
    best_layout = None
    best_score = float('inf')
    
    # First, try single-line layouts in candidate regions
    for region in candidate_regions:
        region_x, region_y, region_width, region_height = region
        
        # Check if region is large enough for all blocks in a single line
        total_width_needed = sum([w + 20 for w, _ in adjusted_block_sizes]) - 20  # Subtract last spacing
        if region_width < total_width_needed:
            continue
            
        # Try placing all blocks in this region in a single line
        layout_blocks = []
        layout_positions = []
        current_x = region_x + 10
        current_y = region_y + 10
        layout_failed = False
        
        for i, (block, (new_width, new_height)) in enumerate(zip(vertical_caption_blocks, adjusted_block_sizes)):
            # Check boundaries
            if current_x + new_width > region_x + region_width - 10:
                layout_failed = True
                break
                
            # Check overlap
            proposed_bbox = (current_x, current_y, current_x + new_width, current_y + new_height)
            if check_overlap(proposed_bbox, existing_bbox_list + layout_positions):
                layout_failed = True
                break
                
            # Create new block
            new_lines = np.array([
                [
                    [current_x, current_y],
                    [current_x + new_width, current_y],
                    [current_x + new_width, current_y + new_height],
                    [current_x, current_y + new_height],
                ]
            ], dtype=np.int32)
            
            new_block = copy.deepcopy(block)
            new_block.lines = new_lines
            new_block._direction = "h"
            new_block.is_rearranged = True
            
            layout_blocks.append(new_block)
            layout_positions.append(proposed_bbox)
            
            # Move to next position
            current_x += new_width + 20
            
        if not layout_failed:
            # Calculate score based on reading order preservation and position
            
            # 1. Reading order score - check if blocks maintain correct sequence
            reading_order_score = 0
            for i in range(len(layout_positions)-1):
                # In single line layout, blocks should be arranged left to right
                # in the same order as original right to left
                if layout_positions[i][0] >= layout_positions[i+1][0]:  # Left to right violated
                    reading_order_score += 10
            
            # 2. Distance from original position (weighted less)
            position_score = 0
            for i, pos in enumerate(layout_positions):
                new_center = ((pos[0] + pos[2])/2, (pos[1] + pos[3])/2)
                original_center = original_centers[i]
                position_score += np.linalg.norm(np.array(new_center) - original_center) * 0.01
            
            # 3. Vertical position score (prefer positions closer to original vertical position)
            vertical_score = 0
            for i, pos in enumerate(layout_positions):
                original_y = original_positions[i][1]
                new_y = pos[1]
                vertical_score += abs(original_y - new_y) * 0.05
            
            # Combined score (reading order is most important)
            total_score = reading_order_score + position_score + vertical_score
            
            if total_score < best_score:
                best_score = total_score
                best_layout = layout_blocks
    
    # If single-line layout didn't work, try multi-line layouts
    if best_layout is None:
        for region in candidate_regions:
            region_x, region_y, region_width, region_height = region
            
            # Skip very narrow regions
            min_block_width = min([w for w, _ in adjusted_block_sizes])
            if region_width < min_block_width + 20:
                continue
                
            # Try multi-line arrangement
            layout_blocks = []
            layout_positions = []
            current_x = region_x + 10
            current_y = region_y + 10
            row_height = 0
            layout_failed = False
            
            for i, (block, (new_width, new_height)) in enumerate(zip(vertical_caption_blocks, adjusted_block_sizes)):
                # Check if we need to move to next line
                if current_x + new_width > region_x + region_width - 10:
                    current_x = region_x + 10
                    current_y += row_height + 10
                    row_height = 0
                    
                # Check if we've exceeded the region height
                if current_y + new_height > region_y + region_height - 10:
                    layout_failed = True
                    break
                
                # Check boundaries
                if current_y + new_height > img_height - 10:
                    layout_failed = True
                    break
                    
                # Check overlap
                proposed_bbox = (current_x, current_y, current_x + new_width, current_y + new_height)
                if check_overlap(proposed_bbox, existing_bbox_list + layout_positions):
                    layout_failed = True
                    break
                    
                # Create new block
                new_lines = np.array([
                    [
                        [current_x, current_y],
                        [current_x + new_width, current_y],
                        [current_x + new_width, current_y + new_height],
                        [current_x, current_y + new_height],
                    ]
                ], dtype=np.int32)
                
                new_block = copy.deepcopy(block)
                new_block.lines = new_lines
                new_block._direction = "h"
                new_block.is_rearranged = True
                
                layout_blocks.append(new_block)
                layout_positions.append(proposed_bbox)
                
                # Update row height and move to next position
                row_height = max(row_height, new_height)
                current_x += new_width + 20
                
            if not layout_failed and len(layout_blocks) == len(vertical_caption_blocks):
                # Calculate score as before but adapted for multi-line layout
                
                # For multi-line layout, we need to consider rows
                rows = []
                current_row = []
                last_y = layout_positions[0][1]
                
                for i, pos in enumerate(layout_positions):
                    if abs(pos[1] - last_y) > 20 and current_row:  # New row detected
                        rows.append(current_row)
                        current_row = [i]
                        last_y = pos[1]
                    else:
                        current_row.append(i)
                
                if current_row:
                    rows.append(current_row)
                
                # Check reading order within and across rows
                reading_order_score = 0
                
                # Within each row, should be left to right
                for row in rows:
                    for i in range(len(row)-1):
                        idx1, idx2 = row[i], row[i+1]
                        if layout_positions[idx1][0] >= layout_positions[idx2][0]:
                            reading_order_score += 5
                
                # Between rows, should be top to bottom
                for i in range(len(rows)-1):
                    top_row_y = layout_positions[rows[i][0]][1]
                    bottom_row_y = layout_positions[rows[i+1][0]][1]
                    if top_row_y >= bottom_row_y:
                        reading_order_score += 10
                
                # Position score as before
                position_score = 0
                for i, pos in enumerate(layout_positions):
                    new_center = ((pos[0] + pos[2])/2, (pos[1] + pos[3])/2)
                    original_center = original_centers[i]
                    position_score += np.linalg.norm(np.array(new_center) - original_center) * 0.01
                
                total_score = reading_order_score + position_score
                
                if total_score < best_score:
                    best_score = total_score
                    best_layout = layout_blocks
    
    # If still no layout found, try default placement
    if best_layout is None:
        logger.warning("Could not find suitable layout in candidate regions. Using fallback layout.")
        best_layout = []
        current_x, current_y = 10, 10
        row_height = 0
        
        for block, (new_width, new_height) in zip(vertical_caption_blocks, adjusted_block_sizes):
            # Simple wrap to next line when reaching edge
            if current_x + new_width > img_width - 10:
                current_x = 10
                current_y += row_height + 10
                row_height = 0
            
            # Create new block
            new_lines = np.array([
                [
                    [current_x, current_y],
                    [current_x + new_width, current_y],
                    [current_x + new_width, current_y + new_height],
                    [current_x, current_y + new_height],
                ]
            ], dtype=np.int32)
            
            new_block = copy.deepcopy(block)
            new_block.lines = new_lines
            new_block._direction = "h"
            new_block.is_rearranged = True
            
            best_layout.append(new_block)
            
            # Update for next placement
            row_height = max(row_height, new_height)
            current_x += new_width + 20
    
    # Combine rearranged vertical blocks with original horizontal blocks
    return best_layout + horizontal_blocks


def check_overlap(
    bbox1: Tuple[int, int, int, int], 
    bbox_list: List[Tuple[int, int, int, int]], 
    margin: int = 10
) -> bool:
    """
    Check if a bounding box overlaps with any box in a list, with added margin.
    
    Args:
        bbox1: Bounding box to check (x1, y1, x2, y2)
        bbox_list: List of existing bounding boxes
        margin: Extra margin to ensure spacing between boxes
        
    Returns:
        True if overlap is detected, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    
    # Add margin for better spacing
    x1_1 -= margin
    y1_1 -= margin
    x2_1 += margin
    y2_1 += margin
    
    for bbox2 in bbox_list:
        x1_2, y1_2, x2_2, y2_2 = bbox2
        if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
            return True
    return False


def visualize_textblocks(canvas: np.ndarray, blk_list: List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for i, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        for j, line in enumerate(blk.lines):
            cv2.putText(
                canvas, str(j), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 127, 0), 1
            )
            cv2.polylines(canvas, [line], True, (0, 127, 255), 2)
        cv2.polylines(canvas, [blk.min_rect], True, (127, 127, 0), 2)
        cv2.putText(
            canvas,
            str(i),
            (bx1, by1 + lw),
            0,
            lw / 3,
            (255, 127, 127),
            max(lw - 1, 1),
            cv2.LINE_AA,
        )
        center = [int((bx1 + bx2) / 2), int((by1 + by2) / 2)]
        cv2.putText(
            canvas,
            "a: %.2f" % blk.angle,
            [bx1, center[1]],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (127, 127, 255),
            2,
        )
        cv2.putText(
            canvas,
            "x: %s" % bx1,
            [bx1, center[1] + 30],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (127, 127, 255),
            2,
        )
        cv2.putText(
            canvas,
            "y: %s" % by1,
            [bx1, center[1] + 60],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (127, 127, 255),
            2,
        )
    return canvas
