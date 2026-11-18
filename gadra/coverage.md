# `coverage.py`

`geometry()` loads captured native Gaussian centers and covariance.
`build_order()` forms bounded candidate priorities using squared center
distance divided by the sum of squared Gaussian scales, then exports a
traversal for dependency-ready grouping.

The original graph determines actual shared inputs. Priority ordering does not
replace convolution, attention, pooling, or recovery dependencies. Group
construction and associated index traffic are included by the replay engine.
