# Task Categorization

## Overview

$n$...set length

$m$...max. element

| Task                     | _Naive Runtime_ | Naive Space | Best Runtime  | Best Space | _Contribution_ | Label Dist. (1) |
| ------------------------ | --------------- | ----------- | ------------- | ---------- | -------------- | --------------- |
| `sum`                    | $n$             | $1$         | $n$           | $1$        | $1$            | N/A             |
| `cardinality`            | $n$             | $1$         | $n$           | $1$        | $1$            | $Uniform(1,L)$  |
| `mode`                   | $n$             | $n$         | N/A           | N/A        | N/A            | N/A             |
| `max`                    | $n$             | $1$         | $n$           | $1$        | $1$            | N/A             |
| `min`                    | $n$             | $1$         | $n$           | $1$        | $1$            | N/A             |
| `mean`                   | $n$             | $n$         | $n$           | $1$        | $1$            | N/A             |
| `longest_seq_length`     | $n^2$           | $n^2$       | $n\log n$     | $1$        | $n$            | N/A             |
| `largest_contiguous_sum` | $n^2$           | $n^2$       | $n$           | $1$        | $n$            | N/A             |
| `largest_pair_sum`       | $n^2$           | $n^2$       | $n$           | $1$        | $n$            | N/A             |
| `largest_triple_sum`     | $n^3$           | $n^3$       | $n$           | $1$        | $n^2$          | N/A             |
| `contains_even`          | $n$             | 1           | $n$           | $1$        | $1$            | N/A             |
| `contains_prime`         | $nm$            | 1           | $n(\log m)^c$ | $1$        | $1$            | N/A             |

**N.B.: All complexities in big-O.**

_(1) assuming that elements are sampled from $Uniform(m,M)$ and set length from $Uniform(1,L)$_

## Proofs