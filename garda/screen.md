# `screen.py`

## External Interface

`replay_trace()` is the in-process fixed-work replay entry used by formal
experiments. Preparation composition helpers merge measured setup work.

## Internal Helpers

Trace simulation and batch screen helpers emit complete accounting without
changing the captured graph.
