# `direct.py`

## External Interface

`supported()` checks direct-stream eligibility and `execute()` performs finite
bank-to-array streaming without an intermediate operand copy.

## Internal Helpers

Translation tables, row products, and masked reads charge directory, bank,
latch, route, and partial-sum resources.
