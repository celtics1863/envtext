from .visualizer_base import VisualizerBase
from ..utils.html_ops import *
import random


class SpanVisualizer(VisualizerBase):    

    TMP_SPANS = """
    <div class="spans" style="line-height: 2.5; direction: {dir}">{content}</div>
    """

    TMP_SPAN = """
    <span style="font-weight: bold; display: inline-block; position: relative; height: {total_height}px;">
        {text}
        {span_slices}
        {span_starts}
    </span>
    """

    TMP_SPAN_SLICE = """
    <span style="background: {bg}; top: {top_offset}px; height: 4px; left: -1px; width: calc(100% + 20px); position: absolute;">
    </span>
    """


    TMP_SPAN_START = """
    <span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 20px); position: absolute;">
        <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
            {label}{kb_link}
        </span>
    </span>
    """

    TMP_SPAN_RTL = """
    <span style="font-weight: bold; display: inline-block; position: relative;">
        {text}
        {span_slices}
        {span_starts}
    </span>
    """

    TMP_SPAN_SLICE_RTL = """
    <span style="background: {bg}; top: {top_offset}px; height: 4px; left: -1px; width: calc(100% + 20px); position: absolute;">
    </span>
    """

    TMP_SPAN_START_RTL = """
    <span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 20px); position: absolute;">
        <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
            {label}{kb_link}
        </span>
    </span>
    """

    # Important: this needs to start with a space!
    TMP_KB_LINK = """
    <a style="text-decoration: none; color: inherit; font-weight: normal" href="{kb_url}">{kb_id}</a>
    """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if "direction" in kwargs and kwargs["direction"] == "rtl":
                self.direction = "rtl"
                self.span_template = self.TMP_SPAN_RTL
                self.span_slice_template = self.TMP_SPAN_SLICE_RTL
                self.span_start_template = self.TMP_SPAN_START_RTL
        else:
            self.direction = "left"
            self.span_template = self.TMP_SPAN
            self.span_slice_template = self.TMP_SPAN_SLICE
            self.span_start_template = self.TMP_SPAN_START

        self.top_offset = kwargs.get("top_offset",40)
        self.span_label_offset  = kwargs.get("span_label_offset",20)
        self.offset_step = kwargs.get("offset_step",17)


    def generate_html(self,tokens,spans,title = "",**kwargs):
        """Render span types in text.
        Spans are rendered per-token, this means that for each token, we check if it's part
        of a span slice (a member of a span type) or a span start (the starting token of a
        given span type).
            tokens (list): Individual tokens in the text
            spans (list): Individual entity spans and their start, end, label, kb_id and kb_url.
            title (str / None): Document title set in Doc.user_data['title'].

        例如：
            tokens = [

            ]
            spans = [
                "start":0,
                "end":5,
                "label":"标签",
                "kb_id":"",
                "kb_url":""
            ]

        """
        per_token_info = []
        # we must sort so that we can correctly describe when spans need to "stack"
        # which is determined by their start token, then span length (longer spans on top),
        # then break any remaining ties with the span label
        spans = sorted(
            spans,
            key=lambda s: (
                s["start_token"],
                -(s["end_token"] - s["start_token"]),
                s["label"],
            ),
        )
        for s in spans:
            # this is the vertical 'slot' that the span will be rendered in
            # vertical_position = span_label_offset + (offset_step * (slot - 1))
            s["render_slot"] = 0
        for idx, token in enumerate(tokens):
            # Identify if a token belongs to a Span (and which) and if it's a
            # start token of said Span. We'll use this for the final HTML render
            token_markup: Dict[str, Any] = {}
            token_markup["text"] = token
            concurrent_spans = 0
            entities = []
            for span in spans:
                ent = {}
                if span["start_token"] <= idx < span["end_token"]:
                    concurrent_spans += 1
                    span_start = idx == span["start_token"]
                    ent["label"] = span["label"]
                    ent["is_start"] = span_start
                    if span_start:
                        # When the span starts, we need to know how many other
                        # spans are on the 'span stack' and will be rendered.
                        # This value becomes the vertical render slot for this entire span
                        span["render_slot"] = concurrent_spans
                    ent["render_slot"] = span["render_slot"]
                    kb_id = span.get("kb_id", "")
                    kb_url = span.get("kb_url", "#")
                    ent["kb_link"] = (
                        self.TMP_KB_LINK.format(kb_id=kb_id, kb_url=kb_url) if kb_id else ""
                    )
                    entities.append(ent)
                else:
                    # We don't specifically need to do this since we loop
                    # over tokens and spans sorted by their start_token,
                    # so we'll never use a span again after the last token it appears in,
                    # but if we were to use these spans again we'd want to make sure
                    # this value was reset correctly.
                    span["render_slot"] = 0
            token_markup["entities"] = entities
            per_token_info.append(token_markup)
        markup = self._render_markup(per_token_info)
        markup = self.TMP_SPANS.format(content=markup, dir=self.direction)
        if title:
            markup = self.TMP_TITLE.format(title=title) + markup
        return markup

    def _render_markup(self, per_token_info):
        """Render the markup from per-token information"""
        markup = ""
        for token in per_token_info:
            entities = sorted(token["entities"], key=lambda d: d["render_slot"])
            # Whitespace tokens disrupt the vertical space (no line height) so that the
            # span indicators get misaligned. We don't render them as individual
            # tokens anyway, so we'll just not display a span indicator either.
            is_whitespace = token["text"].strip() == ""
            if entities and not is_whitespace:
                slices = self._get_span_slices(token["entities"])
                starts = self._get_span_starts(token["entities"])
                total_height = (
                    self.top_offset
                    + self.span_label_offset
                    + (self.offset_step * (len(entities) - 1))
                )
                markup += self.span_template.format(
                    text=token["text"],
                    span_slices=slices,
                    span_starts=starts,
                    total_height=total_height,
                )
            else:
                markup += escape_html(token["text"] + " ")
        return markup

    def _get_span_slices(self, entities):
        """Get the rendered markup of all Span slices"""
        span_slices = []
        for entity in entities:
            # rather than iterate over multiples of offset_step, we use entity['render_slot']
            # to determine the vertical position, since that tells where
            # the span starts vertically so we can extend it horizontally,
            # past other spans that might have already ended
            color = self._get_color(entity["label"])
            top_offset = self.top_offset + (
                self.offset_step * (entity["render_slot"] - 1)
            )
            span_slice = self.span_slice_template.format(
                bg=color,
                top_offset=top_offset,
            )
            span_slices.append(span_slice)
        return "".join(span_slices)



    def _get_span_starts(self, entities):
        """Get the rendered markup of all Span start tokens"""
        span_starts = []
        for entity in entities:
            color = self._get_color(entity["label"])
            top_offset = self.top_offset + (
                self.offset_step * (entity["render_slot"] - 1)
            )
            span_start = (
                self.span_start_template.format(
                    bg=color,
                    top_offset=top_offset,
                    label=entity["label"],
                    kb_link=entity["kb_link"],
                )
                if entity["is_start"]
                else ""
            )
            span_starts.append(span_start)
        return "".join(span_starts)