import math
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
    if True or font_size_minimum == -1:
        # Automatically determine font_size by image size
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    logger.debug(f"font_size_minimum {font_size_minimum}")

    dst_points_list = []
    for region in text_regions:
        char_count_orig = len(region.text)
        char_count_trans = len(region.translation.strip())
        if char_count_trans > char_count_orig:
            # More characters were added, have to reduce fontsize to fit allotted area
            # print('count', char_count_trans, region.font_size)
            rescaled_font_size = region.font_size
            while True:
                rows = region.unrotated_size[0] // rescaled_font_size
                cols = region.unrotated_size[1] // rescaled_font_size
                if rows * cols >= char_count_trans:
                    # print(rows, cols, rescaled_font_size, rows * cols, char_count_trans)
                    # print('rescaled', rescaled_font_size)
                    region.font_size = rescaled_font_size
                    break
                rescaled_font_size -= 1
                if rescaled_font_size <= 0:
                    break
        # Otherwise no need to increase fontsize

        # Infer the target fontsize
        target_font_size = region.font_size
        if font_size_fixed is not None:
            target_font_size = font_size_fixed
        elif target_font_size < font_size_minimum:
            target_font_size = max(region.font_size, font_size_minimum)
        target_font_size += font_size_offset

        # Rescale dst_points accordingly
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
            region.font_size = int(target_font_size)
        else:
            dst_points = region.min_rect

        dst_points_list.append(dst_points)
    return dst_points_list


def is_in_speech_balloon(region: TextBlock, img: np.ndarray) -> bool:
    """
    Determine if the text region (based on its bounding box) is inside a speech balloon.
    Assumes that speech balloons have a high mean brightness and low variance.
    """
    x1, y1, x2, y2 = region.xyxy
    # Clip coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    patch = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    var_val = gray.var()
    # Heuristic: speech balloons tend to be bright with low variance.
    if mean_val > 200 and var_val < 50:
        return True
    return False


def is_expand_needed(region: TextBlock, img: np.ndarray) -> bool:
    # log every elements that can effect the decision
    # logger.debug(f"region {region.translation[:3]}({len(region.translation[:3])}): font size: {region.font_size}, unrotated_size: {region.unrotated_size}")
    
    # Do not expand if region is vertical.
    if region.vertical:
        # logger.debug(f"region {region.translation[:3]} is vertical")
        return False
    # Do not expand for short text (7 or fewer characters).
    if len(region.get_translation_for_rendering()) <= 7:
        # logger.debug(f"region {region.translation[:3]} is short")
        return False
    # Do not expand if the line is already wide enough.
    char_per_line = region.unrotated_size[0] // region.font_size
    if char_per_line > 10:
        # logger.debug(f"region {region.translation[:3]} has enough characters per line - {char_per_line}, font_size {region.font_size}, unrotated_size {region.unrotated_size}")
        return False
    # Do not expand if region is inside a speech balloon.
    if is_in_speech_balloon(region, img):
        # logger.debug(f"region {region.translation[:3]} is in a speech balloon")
        return False
    
    # 텍스트 박스의 모양이 세로로 긴 형태일 때 True 리턴
    if region.unrotated_size[1] > region.unrotated_size[0] * 2:
        # logger.debug(f"region {region.translation[:3]} is long")
        return True

    return False


def expand_text_boxes(
    regions: List[TextBlock], expand_box_width_ratio: float, img: np.ndarray
):
    """
    Expand text boxes that need expansion (as per is_expand_needed) by scaling their width (for horizontal text)
    or height (for vertical text) by expand_box_width_ratio. Adjust positions to avoid overlapping and ensure
    boxes remain within the image boundaries.
    """
    if expand_box_width_ratio <= 0 or math.isclose(expand_box_width_ratio, 1.0):
        return regions  # No change needed

    def boxes_overlap(box1, box2):
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        return not (x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1)

    expanded_regions = []
    placed_boxes = []  # Store already placed (x1, y1, x2, y2) boxes

    for region in regions:
        bbox = region.xyxy  # [x1, y1, x2, y2]
        if not is_expand_needed(region, img):
            placed_boxes.append(tuple(bbox.tolist()))
            expanded_regions.append(region)
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if region.horizontal:
            new_width = int(width * expand_box_width_ratio)
            new_height = height
            center_x = (x1 + x2) // 2
            new_x1 = center_x - new_width // 2
            new_x2 = center_x + new_width // 2
            new_y1, new_y2 = y1, y2
        else:
            new_height = int(height * expand_box_width_ratio)
            new_width = width
            center_y = (y1 + y2) // 2
            new_y1 = center_y - new_height // 2
            new_y2 = center_y + new_height // 2
            new_x1, new_x2 = x1, x2

        # Clip within image boundaries
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img.shape[1], new_x2)
        new_y2 = min(img.shape[0], new_y2)
        proposed_box = (new_x1, new_y1, new_x2, new_y2)

        # If overlapping with any already placed box, shift (right for horizontal, down for vertical)
        shift = 0
        max_shift = 100
        while (
            any(boxes_overlap(proposed_box, placed) for placed in placed_boxes)
            and shift < max_shift
        ):
            shift += 5
            if region.horizontal:
                new_x1_shift = min(new_x1 + shift, img.shape[1] - new_width)
                new_x2_shift = new_x1_shift + new_width
                proposed_box = (new_x1_shift, new_y1, new_x2_shift, new_y2)
            else:
                new_y1_shift = min(new_y1 + shift, img.shape[0] - new_height)
                new_y2_shift = new_y1_shift + new_height
                proposed_box = (new_x1, new_y1_shift, new_x2, new_y2_shift)

        placed_boxes.append(proposed_box)
        # Update region.lines with the new rectangular coordinates
        new_line = np.array(
            [
                [proposed_box[0], proposed_box[1]],
                [proposed_box[2], proposed_box[1]],
                [proposed_box[2], proposed_box[3]],
                [proposed_box[0], proposed_box[3]],
            ],
            dtype=np.int32,
        )
        region.lines = np.array([new_line])
        expanded_regions.append(region)

    return expanded_regions


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
    text_regions = list(filter(lambda region: region.translation.strip(), text_regions))
    
    # logger.debug(f"before expand: {text_regions}")
    text_regions = expand_text_boxes(
        text_regions, config.render.expand_box_width_ratio, img
    )
    # logger.debug(f"after expand: {text_regions}")

    log_text = "\n".join([str(i) for i in text_regions])
    # logger.debug(f"text_regions before rearrange: {log_text}")

    try:
        pass
        # text_regions = rearrange_vertical_text_to_horizontal(text_regions, img)
    except Exception as e:
        logger.error(f"Error while rearranging text: {e}", exc_info=True)
    log_text = "\n".join([str(i) for i in text_regions])

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(
        img, text_regions, font_size_fixed, font_size_offset, font_size_minimum
    )

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(
        zip(text_regions, dst_points_list), "[render]", total=len(text_regions)
    ):
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
        # region.maximize_korean_font_size()
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
    # Modified to keep text at the top-left corner instead of centering it
    if r_temp > r_orig:
        h_ext = int(w / r_orig - h)
        box = np.zeros((h + h_ext, w, 4), dtype=np.uint8)
        box[0:h, 0:w] = temp_box  # Text positioned at the top
    else:
        w_ext = int(h * r_orig - w)
        box = np.zeros((h, w + w_ext, 4), dtype=np.uint8)
        box[0:h, 0:w] = temp_box  # Text positioned at the left

    src_points = np.array(
        [[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]
    ).astype(np.float32)
    # src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    # src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    # Add the missing perspective transform calculation
    M = cv2.getPerspectiveTransform(src_points, dst_points.astype(np.float32))
    # logger.debug(
    #     f"for {region.translation[:3]}, souce points {src_points}, dst_points {dst_points}"
    # )
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
