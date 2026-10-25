# `diagnostic_pipeline.py`

## External Interface

`execute()` lowers A observations onto the shared finite engine. Initialization,
feedback, witness, retirement, and release functions manage its lifecycle.

## Internal Helpers

`Program` emits shaped matrix/vector micro-operations; query, language, and
control helpers account for storage and dependency costs.
