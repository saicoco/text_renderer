from cgitb import grey
import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
    SimpleTextColorCfg,
    AdaptiveTextColorCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout
import numpy as np

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
EVAL_OUT_DIR = CURRENT_DIR / "output_EVAL"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg" / "color_bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(80, 81),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(15, 20),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=False
):
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def get_char_hebrew_corpus():
    return CharCorpusHeBrew(
        CharCorpusCfg(
            # text_paths=[TEXT_DIR / "hebrew_text.txt"],
            text_paths=[TEXT_DIR / "hebrew_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "hebrew.txt",
            length=(10, 25),
            char_spacing=(0, 0.1),
            **font_cfg
        ),
    )


def hebrew_data():

    poses = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus(),
        gray=False,
        layout=SameLineLayout(),
        corpus_effects=Effects(
            [
                Line(0.5, thickness=(3, 4), color_cfg=AdaptiveTextColorCfg()),
            ]
        ),
    )

    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def get_hebrew_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[
                TEXT_DIR / "hebrew_text.txt",
            ],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "hebrew.txt",
            length=(15, 16),
            char_spacing=(-0.3, 1.3),
            text_color_cfg=AdaptiveTextColorCfg(),
            **font_cfg
        ),
    )


def hebrew_data():
    poses = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]
    lines_effects = []
    for i, p in enumerate(poses):
        pos_val = [0] * len(poses)
        pos_val[i] = 1
        lines_effects.append(
            Line(0.5, color_cfg=AdaptiveTextColorCfg(), line_pos_p=pos_val)
        )

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_hebrew_char_corpus(),
        gray=False,
        corpus_effects=Effects(
            [
                OneOf(lines_effects),
                # OneOf([DropoutRand(p=0.2), DropoutVertical(p=0.2)]),
                # Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.2, 0.5], center=True),
            ]
        ),
    )
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def enum_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def hebrew_data_compact_space():

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def hebrew_data_emboss():

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.height = 48
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def extra_text_line_layout():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.layout = ExtraTextLineLayout(bottom_prob=1.0)
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def same_line_layout_different_font_size():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.layout = SameLineLayout(h_spacing=(0.9, 0.91))
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def padding():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(
        Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
    )
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def dropout_rand():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def dropout_horizontal():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


def dropout_vertical():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name, corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    cfg.render_cfg.text_color_cfg = AdaptiveTextColorCfg()
    return cfg


# fmt: off
# The configuration file must have a configs variable
configs = [
    hebrew_data(),
    # hebrew_data_compact_space(),
    # hebrew_data_emboss(),
    # extra_text_line_layout(),
    # same_line_layout_different_font_size(),
    # padding(),
    # dropout_horizontal(),
    # dropout_rand(),
    # dropout_vertical()
]
# fmt: on
