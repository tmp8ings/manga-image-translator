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

        logger.debug(
            f"This is a vertical caption for{self.translation[:3]}: {self.xyxy[1]}, {self.aspect_ratio}"
        )
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


# def sort_textblk_list(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
#     if len(blk_list) == 0:
#         return blk_list
#     num_ja = 0
#     xyxy = []
#     for blk in blk_list:
#         if blk.language == 'ja':
#             num_ja += 1
#         xyxy.append(blk.xyxy)
#     xyxy = np.array(xyxy)
#     flip_lr = num_ja > len(blk_list) / 2
#     im_oriw = im_w
#     if im_w > im_h:
#         im_w /= 2
#     num_gridy, num_gridx = 4, 3
#     img_area = im_h * im_w
#     center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
#     if flip_lr:
#         if im_w != im_oriw:
#             center_x = im_oriw - center_x
#         else:
#             center_x = im_w - center_x
#     grid_x = (center_x / im_w * num_gridx).astype(np.int32)
#     center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
#     grid_y = (center_y / im_h * num_gridy).astype(np.int32)
#     grid_indices = grid_y * num_gridx + grid_x
#     grid_weights = grid_indices * img_area + 1.2 * (center_x - grid_x * im_w / num_gridx) + (center_y - grid_y * im_h / num_gridy)
#     if im_w != im_oriw:
#         grid_weights[np.where(grid_x >= num_gridx)] += img_area * num_gridy * num_gridx

#     for blk, weight in zip(blk_list, grid_weights):
#         blk.sort_weight = weight
#     blk_list.sort(key=lambda blk: blk.sort_weight)
#     return blk_list

# # TODO: Make these cached_properties
# def examine_textblk(blk: TextBlock, im_w: int, im_h: int, sort: bool = False) -> None:
#     lines = blk.lines_array()
#     middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
#     vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors of textlines
#     vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors of textlines
#     # if sum of vertical vectors is longer, then text orientation is vertical, and vice versa.
#     center_pnts = (lines[:, 0] + lines[:, 2]) / 2
#     v = np.sum(vec_v, axis=0)
#     h = np.sum(vec_h, axis=0)
#     norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
#     if blk.language == 'ja':
#         vertical = norm_v > norm_h
#     else:
#         vertical = norm_v > norm_h * 2
#     # calculate distance between textlines and origin
#     if vertical:
#         primary_vec, primary_norm = v, norm_v
#         distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
#         font_size = int(round(norm_h / len(lines)))
#     else:
#         primary_vec, primary_norm = h, norm_h
#         distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
#         font_size = int(round(norm_v / len(lines)))

#     rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)     # rotation angle of textlines
#     distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
#     rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
#     distance = np.abs(np.sin(rad_matrix) * distance)
#     blk.lines = lines.astype(np.int32).tolist()
#     blk.distance = distance
#     blk.angle = rotation_angle
#     if vertical:
#         blk.angle -= 90
#     if abs(blk.angle) < 3:
#         blk.angle = 0
#     blk.font_size = font_size
#     blk.vertical = vertical
#     blk.vec = primary_vec
#     blk.norm = primary_norm
#     if sort:
#         blk.sort_lines()

# def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.4, distance_tol=2) -> bool:
#     if blk2.merged:
#         return False
#     fntsize_div = blk.font_size / blk2.font_size
#     num_l1, num_l2 = len(blk), len(blk2)
#     fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
#     vec_prod = blk.vec @ blk2.vec
#     vec_sum = blk.vec + blk2.vec
#     cos_vec = vec_prod / blk.norm / blk2.norm
#     distance = blk2.distance[-1] - blk.distance[-1]
#     distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
#     l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
#     if not l1.intersects(l2):
#         if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
#             return False
#         if abs(cos_vec) < 0.866:   # cos30
#             return False
#         # if distance > distance_tol * fntsz_avg or distance_p1 > fntsz_avg * 2.5:
#         if distance > distance_tol * fntsz_avg:
#             return False
#         if blk.vertical and blk2.vertical and distance_p1 > fntsz_avg * 2.5:
#             return False
#     # merge
#     blk.lines.append(blk2.lines[0])
#     blk.vec = vec_sum
#     blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
#     if blk.vertical:
#         blk.angle -= 90
#     blk.norm = np.linalg.norm(vec_sum)
#     blk.distance = np.append(blk.distance, blk2.distance[-1])
#     blk.font_size = fntsz_avg
#     blk2.merged = True
#     return True

# def merge_textlines(blk_list: List[TextBlock]) -> List[TextBlock]:
#     if len(blk_list) < 2:
#         return blk_list
#     blk_list.sort(key=lambda blk: blk.distance[0])
#     merged_list = []
#     for ii, current_blk in enumerate(blk_list):
#         if current_blk.merged:
#             continue
#         for jj, blk in enumerate(blk_list[ii+1:]):
#             try_merge_textline(current_blk, blk)
#         merged_list.append(current_blk)
#     for blk in merged_list:
#         blk.adjust_bbox(with_bbox=False)
#     return merged_list

# def split_textblk(blk: TextBlock):
#     font_size, distance, lines = blk.font_size, blk.distance, blk.lines
#     l0 = np.array(blk.lines[0])
#     lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
#     distance_tol = font_size * 2
#     current_blk = copy.deepcopy(blk)
#     current_blk.lines = [l0]
#     sub_blk_list = [current_blk]
#     textblock_splitted = False
#     for jj, line in enumerate(lines[1:]):
#         l1, l2 = Polygon(lines[jj]), Polygon(line)
#         split = False
#         if not l1.intersects(l2):
#             line_disance = abs(distance[jj+1] - distance[jj])
#             if line_disance > distance_tol:
#                 split = True
#             elif blk.vertical and abs(blk.angle) < 15:
#                 if len(current_blk.lines) > 1 or line_disance > font_size:
#                     split = abs(lines[jj][0][1] - line[0][1]) > font_size
#         if split:
#             current_blk = copy.deepcopy(current_blk)
#             current_blk.lines = [line]
#             sub_blk_list.append(current_blk)
#         else:
#             current_blk.lines.append(line)
#     if len(sub_blk_list) > 1:
#         textblock_splitted = True
#         for current_blk in sub_blk_list:
#             current_blk.adjust_bbox(with_bbox=False)
#     return textblock_splitted, sub_blk_list

# def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True) -> List[TextBlock]:
#     blk_list: List[TextBlock] = []
#     scattered_lines = {'ver': [], 'hor': []}
#     for bbox, lang_id, conf in zip(*blks):
#         # cls could give wrong result
#         blk_list.append(TextBlock(bbox, language=LANG_LIST[lang_id]))

#     # step1: filter & assign lines to textblocks
#     bbox_score_thresh = 0.4
#     mask_score_thresh = 0.1
#     for line in lines:
#         bx1, bx2 = line[:, 0].min(), line[:, 0].max()
#         by1, by2 = line[:, 1].min(), line[:, 1].max()
#         bbox_score, bbox_idx = -1, -1
#         line_area = (by2-by1) * (bx2-bx1)
#         for i, blk in enumerate(blk_list):
#             score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
#             if bbox_score < score:
#                 bbox_score = score
#                 bbox_idx = i
#         if bbox_score > bbox_score_thresh:
#             blk_list[bbox_idx].lines.append(line)
#         else: # if no textblock was assigned, check whether there is "enough" textmask
#             if mask is not None:
#                 mask_score = mask[by1: by2, bx1: bx2].mean() / 255
#                 if mask_score < mask_score_thresh:
#                     continue
#             blk = TextBlock([bx1, by1, bx2, by2], [line])
#             examine_textblk(blk, im_w, im_h, sort=False)
#             if blk.vertical:
#                 scattered_lines['ver'].append(blk)
#             else:
#                 scattered_lines['hor'].append(blk)

#     # step2: filter textblocks, sort & split textlines
#     final_blk_list = []
#     for blk in blk_list:
#         # filter textblocks
#         if len(blk.lines) == 0:
#             bx1, by1, bx2, by2 = blk.xyxy
#             if mask is not None:
#                 mask_score = mask[by1: by2, bx1: bx2].mean() / 255
#                 if mask_score < mask_score_thresh:
#                     continue
#             xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
#             blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
#         examine_textblk(blk, im_w, im_h, sort=True)

#         # split manga text if there is a distance gap
#         textblock_splitted = False
#         if len(blk.lines) > 1:
#             if blk.language == 'ja':
#                 textblock_splitted = True
#             elif blk.vertical:
#                 textblock_splitted = True
#         if textblock_splitted:
#             textblock_splitted, sub_blk_list = split_textblk(blk)
#         else:
#             sub_blk_list = [blk]
#         # modify textblock to fit its textlines
#         if not textblock_splitted:
#             for blk in sub_blk_list:
#                 blk.adjust_bbox(with_bbox=True)
#         final_blk_list += sub_blk_list

#     # step3: merge scattered lines, sort textblocks by "grid"
#     final_blk_list += merge_textlines(scattered_lines['hor'])
#     final_blk_list += merge_textlines(scattered_lines['ver'])
#     if sort_blklist:
#         final_blk_list = sort_textblk_list(final_blk_list, im_w, im_h)

#     for blk in final_blk_list:
#         if blk.language != 'ja' and not blk.vertical:
#             num_lines = len(blk.lines)
#             if num_lines == 0:
#                 continue
#             # blk.line_spacing = blk.bounding_rect()[3] / num_lines / blk.font_size
#             expand_size = max(int(blk.font_size * 0.1), 3)
#             rad = np.deg2rad(blk.angle)
#             shifted_vec = np.array([[[-1, -1],[1, -1],[1, 1],[-1, 1]]])
#             shifted_vec = shifted_vec * np.array([[[np.sin(rad), np.cos(rad)]]]) * expand_size
#             lines = blk.lines_array() + shifted_vec
#             lines[..., 0] = np.clip(lines[..., 0], 0, im_w-1)
#             lines[..., 1] = np.clip(lines[..., 1], 0, im_h-1)
#             blk.lines = lines.astype(np.int64).tolist()
#             blk.font_size += expand_size

#     return final_blk_list


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


def blocks_overlap(rect1, rect2):
    """
    Check if two rectangles (x1, y1, x2, y2) overlap
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Check if one rectangle is to the left of the other
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    # Check if one rectangle is above the other
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True


def rearrange_vertical_text_to_horizontal(
    text_blocks: List[TextBlock], img: np.ndarray
) -> List[TextBlock]:
    """
    Rearrange vertical caption text to horizontal positions in the image

    Args:
        text_blocks: List of TextBlock objects
        img: Source image

    Returns:
        List[TextBlock]: List of TextBlocks after rearrangement
    """
    # Separate vertical caption and horizontal blocks
    vertical_caption_blocks = [
        blk for blk in text_blocks if blk.is_vertical_caption(img)
    ]
    horizontal_blocks = [blk for blk in text_blocks if not blk.is_vertical_caption(img)]

    if not vertical_caption_blocks:
        return text_blocks  # No vertical captions to rearrange

    # Sort vertical captions by reading order (top to bottom, right to left)
    vertical_caption_blocks.sort(key=lambda blk: (blk.center[0], blk.center[1]))

    logger.debug(
        f"Found {len(vertical_caption_blocks)} vertical caption blocks to rearrange"
    )

    # Find candidate locations for placement
    candidate_locations = find_candidate_locations(img, horizontal_blocks)
    logger.debug(f"Found {len(candidate_locations)} candidate locations for placement")

    # Process each vertical caption block
    result_blocks = horizontal_blocks.copy()
    for vcap_block in vertical_caption_blocks:
        # Change direction to horizontal
        vcap_block._direction = "h"

        # Calculate appropriate dimensions for horizontal text
        text_length = (
            len(vcap_block.translation)
            if vcap_block.translation
            else len(vcap_block.text)
        )

        # Find best placement location
        best_location = find_best_placement(
            vcap_block, candidate_locations, result_blocks, img, text_length
        )

        # Update block position if we found a location
        if best_location:
            # Calculate new lines based on best_location
            new_lines = create_horizontal_lines(vcap_block, best_location)
            vcap_block.lines = new_lines
            vcap_block.is_rearranged = True  # Mark as rearranged

            # Add this rearranged block to our result list
            result_blocks.append(vcap_block)
            logger.debug(
                f"Rearranged block: {vcap_block.translation[:10]}... to {best_location}"
            )
        else:
            logger.debug(
                f"Could not find placement for block: {vcap_block.translation[:10]}..."
            )

    # Return all blocks (both original horizontal and rearranged vertical)
    return result_blocks + [
        blk for blk in vertical_caption_blocks if not blk.is_rearranged
    ]


def find_candidate_locations(
    img: np.ndarray, existing_blocks: List[TextBlock]
) -> List[Tuple[int, int, int, int]]:
    """
    Find candidate locations for text block placement based on image background

    Args:
        img: Source image
        existing_blocks: List of existing text blocks

    Returns:
        List of (x1, y1, x2, y2) tuples representing candidate locations
    """
    height, width = img.shape[:2]

    # Create a mask of existing blocks with padding
    padding = 10  # Add padding around existing blocks
    mask = np.zeros((height, width), dtype=np.uint8)
    for block in existing_blocks:
        x1, y1, x2, y2 = block.xyxy
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        mask[y1:y2, x1:x2] = 255

    # Potential candidate regions
    candidates = []

    # Check top and bottom regions (divided into segments)
    segment_width = width // 3

    # Top regions
    for i in range(3):
        x1 = i * segment_width
        x2 = (i + 1) * segment_width
        y1 = 0
        y2 = height // 4  # Top quarter of the image

        # Check if this region is suitable (low variance, few existing blocks)
        region = img[y1:y2, x1:x2]
        mask_region = mask[y1:y2, x1:x2]

        if is_suitable_region(region, mask_region):
            candidates.append((x1, y1, x2, y2))

    # Bottom regions
    for i in range(3):
        x1 = i * segment_width
        x2 = (i + 1) * segment_width
        y1 = height - height // 4  # Bottom quarter
        y2 = height

        region = img[y1:y2, x1:x2]
        mask_region = mask[y1:y2, x1:x2]

        if is_suitable_region(region, mask_region):
            candidates.append((x1, y1, x2, y2))

    # Add entire top and bottom regions as fallbacks
    candidates.append((0, 0, width, height // 6))  # Top
    candidates.append((0, height - height // 6, width, height))  # Bottom

    return candidates


def is_suitable_region(img_region: np.ndarray, mask_region: np.ndarray) -> bool:
    """
    Determine if a region is suitable for text placement

    Args:
        img_region: Image region
        mask_region: Mask indicating occupied areas

    Returns:
        bool: True if region is suitable for text placement
    """
    if img_region.size == 0:
        return False

    # Check if region is already mostly occupied
    if np.mean(mask_region) > 15:  # More than ~6% occupied
        return False

    # Convert to grayscale if needed
    if len(img_region.shape) > 2:
        gray_region = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = img_region

    # Calculate variance to check for uniform background
    variance = np.var(gray_region)

    # Check edge density using Canny edge detector
    edges = cv2.Canny(gray_region, 50, 150)
    edge_density = np.mean(edges) / 255.0

    # Region is suitable if it has relatively low variance and low edge density
    return variance < 1200 and edge_density < 0.15


def find_best_placement(
    block: TextBlock,
    candidates: List[Tuple[int, int, int, int]],
    existing_blocks: List[TextBlock],
    img: np.ndarray,
    text_length: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Find the best placement location for a text block

    Args:
        block: TextBlock to place
        candidates: List of candidate locations
        existing_blocks: List of existing text blocks
        img: Source image
        text_length: Length of text for width calculation

    Returns:
        Tuple (x1, y1, x2, y2) representing the best location to place the block
    """
    # Get original position and dimensions
    orig_x, orig_y = block.center
    original_width, original_height = block.unrotated_size
    
    # Calculate the original area - we'll preserve this
    original_area = original_width * original_height
    
    # Step 1: Calculate new width based on text length and other factors
    if text_length > 0:
        # Use text length to estimate width
        char_width = 15  # Pixels per character (adjustable)
        new_width = text_length * char_width
        
        # Ensure it's not too small or too large
        new_width = max(new_width, original_width * 0.8)
        new_width = min(new_width, img.shape[1] * 0.8)  # Don't exceed 80% of image width
    else:
        # If no text length info, estimate based on vertical caption characteristics
        # Vertical captions typically have small width and large height
        # When converted to horizontal, they should be wider and less tall
        new_width = original_height * 1.2  # Use height as a basis for new width
    
    # Step 2: Calculate height to preserve original area
    new_height = original_area / new_width
    
    # Apply reasonable constraints
    new_width = int(max(new_width, 80))  # Minimum width
    new_width = int(min(new_width, img.shape[1] - 20))  # Maximum width
    
    new_height = int(max(new_height, 20))  # Minimum height
    new_height = int(min(new_height, 100, img.shape[0] * 0.2))  # Maximum height
    
    # Recalculate to ensure area preservation after constraints
    adjusted_area = new_width * new_height
    area_ratio = original_area / adjusted_area
    
    # If area changed significantly due to constraints, adjust as needed
    if 0.8 <= area_ratio <= 1.25:
        # Area is reasonably preserved, no need to adjust further
        block_width, block_height = new_width, new_height
    else:
        # Area changed too much, recalculate one dimension
        if area_ratio > 1:  # New area is smaller than original
            # Increase height to compensate (width is often more important)
            block_width = new_width
            block_height = int(original_area / block_width)
            block_height = min(block_height, 100, img.shape[0] * 0.2)  # Apply max limit again
        else:  # New area is larger than original
            # Adjust width to match original area
            block_height = new_height
            block_width = int(original_area / block_height)
            block_width = min(block_width, img.shape[1] - 20)  # Apply max limit again
    
    logger.debug(f"Original dimensions: {original_width}x{original_height}, area: {original_area}")
    logger.debug(f"New dimensions: {block_width}x{block_height}, area: {block_width*block_height}")

    # Score each candidate location
    best_score = float("inf")
    best_location = None

    img_height, img_width = img.shape[:2]

    # If no candidates or very few, add some default positions
    if len(candidates) < 3:
        # Add top edge
        candidates.append((0, 0, img_width, img_height // 6))
        # Add bottom edge
        candidates.append((0, img_height - img_height // 6, img_width, img_height))

    for x1, y1, x2, y2 in candidates:
        # Check if the block fits in this region
        if x2 - x1 < block_width or y2 - y1 < block_height:
            continue

        # Try different positions within the candidate region
        step_y = max(1, int(block_height // 2))
        step_x = max(1, int(block_width // 3))

        for test_y in range(y1, y2 - block_height + 1, step_y):
            for test_x in range(x1, x2 - block_width + 1, step_x):
                # Create test rectangle
                test_rect = (
                    test_x,
                    test_y,
                    test_x + block_width,
                    test_y + block_height,
                )

                # Check for overlaps with existing blocks
                if any(
                    blocks_overlap(test_rect, existing.xyxy)
                    for existing in existing_blocks
                ):
                    continue

                # Calculate score based on distance from original position
                distance_score = np.sqrt(
                    (test_x - orig_x) ** 2 + (test_y - orig_y) ** 2
                )

                # Prefer keeping text at similar vertical level (for reading flow)
                vertical_penalty = abs(test_y - orig_y) * 2

                # Prefer positions closer to the original horizontal position
                horizontal_penalty = abs(test_x - orig_x)

                # Prefer top or bottom of the image (typical caption locations)
                edge_preference = min(test_y, img_height - (test_y + block_height))

                total_score = (
                    distance_score
                    + vertical_penalty
                    + horizontal_penalty
                    - edge_preference * 0.5
                )

                if total_score < best_score:
                    best_score = total_score
                    best_location = test_rect

    # If no location found in candidates, force placement at top or bottom
    if best_location is None:
        # Try top of image
        top_rect = (
            10,
            10,
            min(10 + block_width, img_width - 10),
            min(10 + block_height, img_height // 3),
        )
        if not any(
            blocks_overlap(top_rect, existing.xyxy) for existing in existing_blocks
        ):
            return top_rect

        # Try bottom of image
        bottom_rect = (
            10,
            max(10, img_height - block_height - 10),
            min(10 + block_width, img_width - 10),
            img_height - 10,
        )
        if not any(
            blocks_overlap(bottom_rect, existing.xyxy) for existing in existing_blocks
        ):
            return bottom_rect

        # Last resort: adjust width and try again with minimal width
        min_width = min(block_width, 200)
        min_height = int(original_area / min_width)  # Maintain area
        min_height = min(min_height, 100)  # Cap the height
        min_rect = (10, 10, 10 + min_width, 10 + min_height)
        return min_rect

    return best_location


def create_horizontal_lines(
    block: TextBlock, new_location: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Create new lines for the block in its new horizontal position

    Args:
        block: TextBlock to modify
        new_location: (x1, y1, x2, y2) for the new location

    Returns:
        np.ndarray: Array of lines in the new location
    """
    x1, y1, x2, y2 = new_location

    # Create a single rectangular line with 4 points (quadrilateral)
    new_line = np.array(
        [
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ],
        dtype=np.int32,
    )

    # Return as a single-element array with the correct shape
    return np.array([new_line], dtype=np.int32)
