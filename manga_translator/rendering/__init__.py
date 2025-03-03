import os
import cv2
from manga_translator.config import Config
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from manga_translator.utils.textblock import rearrange_vertical_text_to_horizontal

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger("render")


def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(",")
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed


def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg


def resize_regions_to_font_size(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_size_fixed: int,
    font_size_offset: int,
    font_size_minimum: int,
):
    """
    더 향상된 폰트 크기 계산 함수
    텍스트 길이, 텍스트박스 크기, 이미지 크기를 고려하여 최적의 폰트 크기 결정
    """
    # 이미지 크기에 따른 기본 최소 폰트 크기 계산 (더 큰 값으로 설정)
    if font_size_minimum == -1 or True:  # 항상 자동 계산 적용
        # 이미지 크기에 따른 최소 폰트 크기 계산식 개선
        img_size_factor = (img.shape[0] + img.shape[1]) / 150  # 분모를 작게 하여 기본 크기 증가
        font_size_minimum = max(round(img_size_factor), 18)  # 최소 18 이상
    
    logger.debug(f"font_size_minimum {font_size_minimum}")
    
    # 글로벌 폰트 스케일 팩터 (전체적인 폰트 크기 증가)
    global_scale_factor = 1.3  # 30% 증가

    dst_points_list = []
    for region in text_regions:
        char_count_orig = len(region.text) if region.text else 1
        char_count_trans = len(region.translation.strip()) if region.translation else 1
        
        # 텍스트 박스 면적 계산
        box_area = region.unrotated_size[0] * region.unrotated_size[1]
        
        # 글자당 이상적인 면적 계산 (더 넓게 설정)
        ideal_area_per_char = box_area / char_count_orig * 1.2
        
        # 번역 텍스트를 위한 이상적인 폰트 크기 계산
        ideal_font_size = (ideal_area_per_char / 2)**0.5  # 제곱근으로 면적->길이 변환
        
        # 텍스트 길이와 원본 폰트 크기 관계 고려
        target_font_size = region.font_size
        
        # 번역된 텍스트가 원본보다 많으면 폰트 크기 조정
        if char_count_trans > char_count_orig:
            compression_ratio = (char_count_orig / char_count_trans) ** 0.5  # 제곱근으로 완화된 비율
            target_font_size = max(
                region.font_size * compression_ratio,
                font_size_minimum
            )
        else:
            # 번역 텍스트가 짧으면 폰트 크기를 키움
            expansion_ratio = min(1.5, (char_count_orig / char_count_trans) ** 0.5)
            target_font_size = region.font_size * expansion_ratio
            
            # 짧은 텍스트 (1-5 글자)는 박스를 더 크게 채워도 좋음
            if char_count_trans <= 5:
                target_font_size *= 1.2
        
        # 계산된 이상적인 폰트 크기와 비교하여 조정
        target_font_size = max(target_font_size, ideal_font_size, font_size_minimum)
        
        # 고정 폰트 크기가 설정되어 있는 경우
        if font_size_fixed is not None:
            target_font_size = font_size_fixed
        # 아니면 최소값과 비교
        elif target_font_size < font_size_minimum:
            target_font_size = font_size_minimum
            
        # 오프셋 적용 및 글로벌 스케일 적용
        target_font_size = target_font_size * global_scale_factor + font_size_offset
        
        # 텍스트 박스 크기 제한 (너무 큰 폰트로 텍스트박스를 넘어가지 않도록)
        # 가로 방향 텍스트이면 높이의 절반을 넘지 않도록, 세로면 너비의 절반을 넘지 않도록
        if region.horizontal:
            max_font_size = region.unrotated_size[1] / 2
        else:
            max_font_size = region.unrotated_size[0] / 2
            
        target_font_size = min(target_font_size, max_font_size)
        
        # 정수로 변환
        target_font_size = int(round(target_font_size))
        
        logger.debug(f"Region: '{region.translation[:10]}...', Original size: {region.font_size}, Target size: {target_font_size}")

        # 폰트 크기가 변경되었으면 점 위치 조정
        if target_font_size != region.font_size:
            target_scale = target_font_size / region.font_size
            dst_points = region.unrotated_min_rect[0]
            poly = Polygon(region.unrotated_min_rect[0])
            poly = affinity.scale(poly, xfact=target_scale, yfact=target_scale)
            dst_points = np.array(poly.exterior.coords[:4])
            dst_points = rotate_polygons(
                region.center, dst_points.reshape(1, -1), -region.angle
            ).reshape(-1, 4, 2)

            # Clip to img width and height
            dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1])
            dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0])

            dst_points = dst_points.reshape((-1, 4, 2))
            region.font_size = target_font_size
        else:
            dst_points = region.min_rect

        dst_points_list.append(dst_points)
    return dst_points_list


async def dispatch(
    config: Config,
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = "",
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False,
) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    log_text = "\n".join([str(i) for i in text_regions])
    logger.debug(f"text_regions before rearrange: {log_text}")
    try:
        text_regions = rearrange_vertical_text_to_horizontal(text_regions, img)
    except Exception as e:
        logger.error(f"Error while rearranging text: {e}", exc_info=True)
    log_text = "\n".join([str(i) for i in text_regions])
    logger.debug(f"text_regions after rearrange: {log_text}")

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(
        img, text_regions, font_size_fixed, font_size_offset, font_size_minimum
    )

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(
        zip(text_regions, dst_points_list), "[render]", total=len(text_regions)
    ):
        if config.render.alignment:
            region._alignment = config.render.alignment
        if config.render.direction:
            region._direction = config.render.direction
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(
            img, region, dst_points, hyphenate, line_spacing, disable_font_border
        )
    return img


def render(
    img, region: TextBlock, dst_points, hyphenate, line_spacing, disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border:
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    # logger.debug(
    #     f"norm_h {norm_h}, norm_v {norm_v}, r_orig {r_orig}, region {region}:{region.horizontal}"
    # )
    if region.horizontal:
        # logger.debug all arguments
        # logger.debug(
        #     f'font_size {region.font_size}, translation {region.get_translation_for_rendering()[:3]}, norm_h {norm_h[0]}, norm_v {norm_v[0]}, alignment {region.alignment}, direction {region.direction == "hr"}, fg {fg}, bg {bg}, target_lang {region.target_lang}, hyphenate {hyphenate}, line_spacing {line_spacing}'
        # )
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == "hr",
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    logger.debug(f"temp_box shape {temp_box.shape}")
    h, w, _ = temp_box.shape
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    if r_temp > r_orig:
        h_ext = int(w / (2 * r_orig) - h / 2)
        box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
        box[h_ext : h + h_ext, 0:w] = temp_box
    else:
        w_ext = int((h * r_orig - w) / 2)
        box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        box[0:h, w_ext : w_ext + w] = temp_box

    src_points = np.array(
        [[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]
    ).astype(np.float32)
    # src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    # src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(
        box,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y : y + h, x : x + w, :3]
    mask_region = rgba_region[y : y + h, x : x + w, 3:4].astype(np.float32) / 255.0
    img[y : y + h, x : x + w] = np.clip(
        (
            img[y : y + h, x : x + w].astype(np.float32) * (1 - mask_region)
            + canvas_region.astype(np.float32) * mask_region
        ),
        0,
        255,
    ).astype(np.uint8)
    return img


async def dispatch_eng_render(
    img_canvas: np.ndarray,
    original_img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = "",
    line_spacing: int = 0,
    disable_font_border: bool = False,
) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, "fonts/comic shanns 2.ttf")
    text_render.set_font(font_path)

    return render_textblock_list_eng(
        img_canvas,
        text_regions,
        line_spacing=line_spacing,
        size_tol=1.2,
        original_img=original_img,
        downscale_constraint=0.8,
        disable_font_border=disable_font_border,
    )
