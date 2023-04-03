from .visualizer_base import VisualizerBase
from ..utils.html_ops import *

class DependencyVisualizer(VisualizerBase):
    """Render dependency parses as SVGs."""

    style = "dep"

    TMP_DEP_SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="{lang}" id="{id}" class="displacy" width="{width}" height="{height}" direction="{dir}" style="max-width: none; height: {height}px; color: {color}; background: {bg}; font-family: {font}; direction: {dir}">{content}</svg>
    """


    TMP_DEP_WORDS = """
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="{y}">
        <tspan class="displacy-word" fill="currentColor" x="{x}">{text}</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="{x}">{tag}</tspan>
    </text>
    """


    TMP_DEP_WORDS_LEMMA = """
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="{y}">
        <tspan class="displacy-word" fill="currentColor" x="{x}">{text}</tspan>
        <tspan class="displacy-lemma" dy="2em" fill="currentColor" x="{x}">{lemma}</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="{x}">{tag}</tspan>
    </text>
    """


    TMP_DEP_ARCS = """
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-{id}-{i}" stroke-width="{stroke}px" d="{arc}" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-{id}-{i}" class="displacy-label" startOffset="50%" side="{label_side}" fill="currentColor" text-anchor="middle">{label}</textPath>
        </text>
        <path class="displacy-arrowhead" d="{head}" fill="currentColor"/>
    </g>
    """

    def __init__(self, *args,**kwargs) -> None:
        """Initialise dependency renderer.
        kwargs (dict): Visualiser-specific options (compact, word_spacing,
            arrow_spacing, arrow_width, arrow_stroke, distance, offset_x,
            color, bg, font)
        """
        super().__init__(*args,**kwargs)

        self.compact = kwargs.get("compact", True)
        self.word_spacing = kwargs.get("word_spacing", 45)
        self.arrow_spacing = kwargs.get("arrow_spacing", 12 if self.compact else 20)
        self.arrow_width = kwargs.get("arrow_width", 6 if self.compact else 10)
        self.arrow_stroke = kwargs.get("arrow_stroke", 2)
        self.distance = kwargs.get("distance", 150 if self.compact else 175)
        self.offset_x = kwargs.get("offset_x", 50)
        self.color = kwargs.get("color", "#000000")
        self.bg = kwargs.get("bg", "#ffffff")
        self.font = kwargs.get("font", "Arial")
        self.direction = "left"
        self.lang = "zh"

    def generate_html(self,render_id,words,arcs,**kwargs):
        """Render SVG.
        render_id (Union[int, str]): Unique ID, typically index of document.
        words (list): Individual words and their tags.
        arcs (list): Individual arcs and their start, end, direction and label.
        RETURNS (str): Rendered SVG markup.
        """
        self.levels = self.get_levels(arcs)
        self.highest_level = max(self.levels.values(), default=0)
        self.offset_y = self.distance / 2 * self.highest_level + self.arrow_stroke
        self.width = self.offset_x + len(words) * self.distance
        self.height = self.offset_y + 3 * self.word_spacing
        self.id = render_id
        words_svg = [
            self.render_word(w["text"], w["tag"], w.get("lemma", None), i)
            for i, w in enumerate(words)
        ]
        arcs_svg = [
            self.render_arrow(a["label"], a["start"], a["end"], a["dir"], i)
            for i, a in enumerate(arcs)
        ]
        content = "".join(words_svg) + "".join(arcs_svg)
        return self.TMP_DEP_SVG.format(
            id=self.id,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )

    def render_word(self, text, tag, lemma, i):
        """Render individual word.
        text (str): Word text.
        tag (str): Part-of-speech tag.
        i (int): Unique ID, typically word index.
        RETURNS (str): Rendered SVG markup.
        """
        y = self.offset_y + self.word_spacing
        x = self.offset_x + i * self.distance
        if self.direction == "rtl":
            x = self.width - x
        html_text = escape_html(text)
        if lemma is not None:
            return self.TMP_DEP_WORDS_LEMMA.format(
                text=html_text, tag=tag, lemma=lemma, x=x, y=y
            )
        return self.TMP_DEP_WORDS.format(text=html_text, tag=tag, x=x, y=y)

    def render_arrow(self, label, start, end, direction, i) :
        """Render individual arrow.
        label (str): Dependency label.
        start (int): Index of start word.
        end (int): Index of end word.
        direction (str): Arrow direction, 'left' or 'right'.
        i (int): Unique ID, typically arrow index.
        RETURNS (str): Rendered SVG markup.
        """
        if start < 0 or end < 0:
            error_args = dict(start=start, end=end, label=label, dir=direction)
            raise ValueError(Errors.E157.format(**error_args))
        level = self.levels[(start, end, label)]
        x_start = self.offset_x + start * self.distance + self.arrow_spacing
        if self.direction == "rtl":
            x_start = self.width - x_start
        y = self.offset_y
        x_end = (
            self.offset_x
            + (end - start) * self.distance
            + start * self.distance
            - self.arrow_spacing * (self.highest_level - level) / 4
        )
        if self.direction == "rtl":
            x_end = self.width - x_end
        y_curve = self.offset_y - level * self.distance / 2
        if self.compact:
            y_curve = self.offset_y - level * self.distance / 6
        if y_curve == 0 and max(self.levels.values(), default=0) > 5:
            y_curve = -self.distance
        arrowhead = self.get_arrowhead(direction, x_start, y, x_end)
        arc = self.get_arc(x_start, y, y_curve, x_end)
        label_side = "right" if self.direction == "rtl" else "left"
        return self.TMP_DEP_ARCS.format(
            id=self.id,
            i=i,
            stroke=self.arrow_stroke,
            head=arrowhead,
            label=label,
            label_side=label_side,
            arc=arc,
        )

    def get_arc(self, x_start, y, y_curve, x_end):
        """Render individual arc.
        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (str): Definition of the arc path ('d' attribute).
        """
        template = "M{x},{y} C{x},{c} {e},{c} {e},{y}"
        if self.compact:
            template = "M{x},{y} {x},{c} {e},{c} {e},{y}"
        return template.format(x=x_start, y=y, c=y_curve, e=x_end)

    def get_arrowhead(self, direction, x, y, end):
        """Render individual arrow head.
        direction (str): Arrow direction, 'left' or 'right'.
        x (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        end (int): X-coordinate of arrow end point.
        RETURNS (str): Definition of the arrow head path ('d' attribute).
        """
        if direction == "left":
            p1, p2, p3 = (x, x - self.arrow_width + 2, x + self.arrow_width - 2)
        else:
            p1, p2, p3 = (end, end + self.arrow_width - 2, end - self.arrow_width + 2)
        return f"M{p1},{y + 2} L{p2},{y - self.arrow_width} {p3},{y - self.arrow_width}"

    def get_levels(self, arcs):
        """Calculate available arc height "levels".
        Used to calculate arrow heights dynamically and without wasting space.
        args (list): Individual arcs and their start, end, direction and label.
        RETURNS (dict): Arc levels keyed by (start, end, label).
        """
        arcs = [dict(t) for t in {tuple(sorted(arc.items())) for arc in arcs}]
        length = max([arc["end"] for arc in arcs], default=0)
        max_level = [0] * length
        levels = {}
        for arc in sorted(arcs, key=lambda arc: arc["end"] - arc["start"]):
            level = max(max_level[arc["start"] : arc["end"]]) + 1
            for i in range(arc["start"], arc["end"]):
                max_level[i] = level
            levels[(arc["start"], arc["end"], arc["label"])] = level
        return levels