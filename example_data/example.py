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
    UniformPerspectiveTransformCfg,
    FixedTextColorDefaultCfg,
    SimpleTextColorHeBrewCfg

)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout
from text_renderer.layout.same_line_hebrew import SameLineLayoutHebrew


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
EVAL_OUT_DIR = CURRENT_DIR / "output_EVAL"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg" / "samplebg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 40),
)

perspective_transform = UniformPerspectiveTransformCfg(20, 20, 1.5)


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=False
):
    return GeneratorCfg(
        num_image=100000,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects
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
            char_spacing=(-0.3, 1.3),
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
                Line(0.5, thickness=(3, 4), color_cfg=SimpleTextColorHeBrewCfg()),
            ]
        ),
    )

    cfg.render_cfg.text_color_cfg = FixedTextColorDefaultCfg()
    return cfg



def hebrew_data_large_space():

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def hebrew_data_compact_space():

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def hebrew_data_emboss():

    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.height = 48
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg





def extra_text_line_layout():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.layout = ExtraTextLineLayout(bottom_prob=1.0)
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def same_line_layout_different_font_size():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.layout = SameLineLayout(h_spacing=(0.9, 0.91))
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg



def padding():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(
        Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
    )
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def dropout_rand():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def dropout_horizontal():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg


def dropout_vertical():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_hebrew_corpus()
    )
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    cfg.render_cfg.text_color_cfg = SimpleTextColorHeBrewCfg()
    return cfg



# fmt: off
# The configuration file must have a configs variable
configs = [
    hebrew_data(),
    hebrew_data_compact_space(),
    hebrew_data_emboss(),
    hebrew_data_large_space(),
    extra_text_line_layout(),
    same_line_layout_different_font_size(),
    padding(),
    dropout_horizontal(),
    dropout_rand(),
    dropout_vertical()
]
# fmt: on
