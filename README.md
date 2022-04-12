# Neural Networks - Set Functions

## Conda Environment Setup

Run the following command in the repository root:

```bash
conda env create -f environment.yml
```

## Cloning Model Code

From the repository root, move to the `deepsets` folder and clone the
[Set Transformer repository](https://github.com/juho-lee/set_transformer):

```bash
cd src/deepsets
git clone https://github.com/juho-lee/set_transformer.git
```

as well as the [FSPool repository](https://github.com/Cyanogenoid/fspool):

```bash
git clone https://github.com/Cyanogenoid/fspool.git
```

# Experiment Settings

## Available Models

| Model                                                                                                                    | model_type              |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| [DeepSets with MLP](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)       | `deepsets_mlp`          |
| [Principal Neighbourhood Aggregation](https://papers.nips.cc/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf) | `pna`                   |
| [Small Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)                                                   | `small_set_transformer` |
| Sorted MLP                                                                                                               | `sorted_mlp`            |

## Available Labels (Tasks)

| Name                    | label                    | Description                                             | Set&rarr;Label            | Multiset&rarr;Label      |
| ----------------------- | ------------------------ | ------------------------------------------------------- | ------------------------- | ------------------------ |
| Sum                     | `sum`                    | Sum of all elements in set                              | {4, 2, 1}&rarr;7          | {4, 2, 1, 1}&rarr;8      |
| Cardinality             | `cardinality`            | Cardinality (size) of the set                           | {4, 2, 1}&rarr;3          | {4, 2, 1, 1}&rarr;4      |
| Mode                    | `mode`                   | Mode value in the Set                                   | NA                        | {4, 2, 1, 1}&rarr;1      |
| Maximum                 | `max`                    | Maximum value in the Set                                | {4, 2, 1}&rarr;4          | {4, 2, 1, 1}&rarr;4      |
| Minimum                 | `min`                    | Minimum value in the set                                | {4, 2, 1}&rarr;1          | {4, 2, 1, 1}&rarr;1      |
| Longest Sequence Length | `longest_seq_length`     | Length of the longest monotonically increasing sequence | {2, 1, 4}&rarr;2          | {2, 1, 1, 4}&rarr;3      |
| Longest Contiguous Sum  | `largest_contiguous_sum` | Largest sum of the subset of the set                    | {2, 1, 4, -10, 7}&rarr;14 | {2, 1, 1, 4, -10}&rarr;8 |
| Largest pair sum        | `largest_pair_sum`       | Largest sum of a pair of numbers in the set             | {2, 1, 4, -10, 7}&rarr;11 | {2, 1, 1, 4, -10}&rarr;6 |
| Largest triplet sum     | `largest_triple_sum`     | Largest sum of a triplet of numbers in the set          | {2, 1, 4, -10, 7}&rarr;13 | {2, 1, 1, 4, -10}&rarr;7 |
| Contains even number    | `contains_even`          | 1 if the set contains an even number, 0 otherwise       | {2, 1, 4, -10, 7}&rarr;1  | {3, 1, 1, 5, -9}&rarr;0  |
