# `progressive_triton.py`

## External Interface

Mapped response, affine merge, indexed-linear, distance, difference, update,
and observation wrappers launch the corresponding Triton kernels.

## Internal Helpers

Private kernels preserve explicit precision and storage boundaries and do not
select exits.
