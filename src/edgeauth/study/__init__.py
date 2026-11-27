"""Benchmarking study: how well does face verification work on edge hardware, and for whom?

The study runs on public data (LFW), never on enrolled users. It is separate from the
verification lab so that matcher accuracy can be measured without the PAD screen, which
would correctly reject LFW's web photographs as non-live captures.
"""
