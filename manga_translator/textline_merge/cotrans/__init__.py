from collections import Counter
import itertools
from typing import List, Set, Tuple
import unicodedata

import cv2
from manga_translator.config import Config
from manga_translator.utils import Quadrilateral, quadrilateral_can_merge_region_coarse
from manga_translator.utils.textblock import TextBlock
import numpy as np
import networkx as nx


def _is_whitespace(ch):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False


def _is_control(ch):
    """Checks whether `chars` is a control character."""
    """Checks whether `chars` is a whitespace character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(ch):
    """Checks whether `chars` is a punctuation character."""
    """Checks whether `chars` is a whitespace character."""
    cp = ord(ch)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False


def split_text_region(
    bboxes: List[Quadrilateral],
    region_indices: Set[int],
    gamma=0.5,
    sigma=2,
    std_threshold=6.0,
) -> List[Set[int]]:
    region_indices = list(region_indices)
    if len(region_indices) == 1:
        # case #1
        return [set(region_indices)]
    if len(region_indices) == 2:
        # case #2
        fs1 = bboxes[region_indices[0]].font_size
        fs2 = bboxes[region_indices[1]].font_size
        fs = max(fs1, fs2)
        if (
            bboxes[region_indices[0]].distance(bboxes[region_indices[1]])
            < (1 + gamma) * fs
            and abs(bboxes[region_indices[0]].angle - bboxes[region_indices[1]].angle)
            < 4 * np.pi / 180
        ):
            return [set(region_indices)]
        else:
            return [set([region_indices[0]]), set([region_indices[1]])]
    # case 3
    G = nx.Graph()
    for idx in region_indices:
        G.add_node(idx)
    for u, v in itertools.combinations(region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="kruskal", data=True)
    edges = sorted(edges, key=lambda a: a[2]["weight"], reverse=True)
    edge_weights = [a[2]["weight"] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in region_indices])
    std = np.std(edge_weights)
    mean = np.mean(edge_weights)
    if (
        edge_weights[0] <= mean + std * sigma
        or edge_weights[0] <= fontsize * (1 + gamma)
    ) and std < std_threshold:
        return [set(region_indices)]
    else:
        if edge_weights[0] - edge_weights[1] < std * sigma and std < std_threshold:
            return [set(region_indices)]
        G = nx.Graph()
        for idx in region_indices:
            G.add_node(idx)
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(
                split_text_region(
                    bboxes,
                    node_set,
                    gamma=gamma,
                    sigma=sigma,
                    std_threshold=std_threshold,
                )
            )
        return ans
    pass


def count_valuable_text(text):
    return sum(
        [
            1
            for ch in text
            if not _is_punctuation(ch)
            and not _is_control(ch)
            and not _is_whitespace(ch)
        ]
    )


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    box = np.array(box)
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - startidx, 0)
    box = np.array(box)
    return box


def merge_bboxes_text_region(
    bboxes: List[Quadrilateral], width, height, gamma=0.5, sigma=2, std_threshold=6.0
):
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)
        bboxes[i].assigned_index = i
    # step 1: roughly divide into multiple text region candidates
    for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
        if quadrilateral_can_merge_region_coarse(ubox, vbox):
            G.add_edge(u, v)

    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
        # step 2: split each region
        region_indices.extend(
            split_text_region(bboxes, node_set, gamma, sigma, std_threshold)
        )

    for node_set in region_indices:
        nodes = list(node_set)
        # get overall bbox
        txtlns = np.array(bboxes)[nodes]
        kq = np.concatenate([x.pts for x in txtlns], axis=0)
        if sum([int(a.is_approximate_axis_aligned) for a in txtlns]) > len(txtlns) // 2:
            max_coord = np.max(kq, axis=0)
            min_coord = np.min(kq, axis=0)
            merged_box = np.maximum(
                np.array(
                    [
                        np.array([min_coord[0], min_coord[1]]),
                        np.array([max_coord[0], min_coord[1]]),
                        np.array([max_coord[0], max_coord[1]]),
                        np.array([min_coord[0], max_coord[1]]),
                    ]
                ),
                0,
            )
            bbox = np.concatenate([a[None, :] for a in merged_box], axis=0).astype(int)
        else:
            # TODO: use better method
            bbox = np.concatenate(
                [a[None, :] for a in get_mini_boxes(kq)], axis=0
            ).astype(int)
        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in [bboxes[i] for i in nodes]]))
        fg_g = round(np.mean([box.fg_g for box in [bboxes[i] for i in nodes]]))
        fg_b = round(np.mean([box.fg_b for box in [bboxes[i] for i in nodes]]))
        bg_r = round(np.mean([box.bg_r for box in [bboxes[i] for i in nodes]]))
        bg_g = round(np.mean([box.bg_g for box in [bboxes[i] for i in nodes]]))
        bg_b = round(np.mean([box.bg_b for box in [bboxes[i] for i in nodes]]))
        # majority vote for direction
        dirs = [box.direction for box in [bboxes[i] for i in nodes]]
        majority_dir = Counter(dirs).most_common(1)[0][0]
        # sort
        if majority_dir == "h":
            nodes = sorted(
                nodes, key=lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2
            )
        elif majority_dir == "v":
            nodes = sorted(nodes, key=lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
        # yield overall bbox and sorted indices
        yield bbox, nodes, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b


async def run_merge(
    textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False
) -> List[TextBlock]:
    text_regions: List[TextBlock] = []
    for (
        poly_regions,
        textline_indices,
        majority_dir,
        fg_r,
        fg_g,
        fg_b,
        bg_r,
        bg_g,
        bg_b,
    ) in merge_bboxes_text_region(textlines, width, height):
        # Use list of texts instead of manually concatenating them.
        texts = [textlines[idx].text for idx in textline_indices]
        logprob_lengths = [
            (np.log(textlines[idx].prob), len(textlines[idx].text))
            for idx in textline_indices
        ]
        total_logprobs = sum(lp * l for lp, l in logprob_lengths) / sum(l for _, l in logprob_lengths)
        vc = count_valuable_text("".join(texts))
        if vc > 1:
            txtlns = [textlines[i] for i in textline_indices]
            font_size = int(min(txtln.font_size for txtln in txtlns))
            max_font_size = int(max(txtln.font_size for txtln in txtlns))
            angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
            if abs(angle) < 3:
                angle = 0
            lines = [txtln.pts for txtln in txtlns]
            region = TextBlock(
                lines, texts, font_size=font_size, angle=angle, prob=np.exp(total_logprobs),
                fg_color=(fg_r, fg_g, fg_b), bg_color=(bg_r, bg_g, bg_b)
            )
            region.assigned_direction = majority_dir
            region.textlines = txtlns
            text_regions.append(region)
    return text_regions
