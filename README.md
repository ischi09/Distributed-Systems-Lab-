# Distributed Systems Lab

Additionally to installing requirements.txt via conda, you need to run the following command:

```
python -m pip install hydra-core
```

as well as clone the [Set Transformer repository](https://github.com/juho-lee/set_transformer)
_when inside the `deepsets` folder_:

```
cd src/deepsets
git clone https://github.com/juho-lee/set_transformer.git
```

# Experiment Settings

## Available Models

| Model                                                                                                              | model_type     |
| ------------------------------------------------------------------------------------------------------------------ | -------------- |
| [DeepSets with MLP](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | `deepsets_mlp` |

## Available Labels (Tasks)

| Name                    | label                | Description                                             | Set, Label      | Multiset, Label |
| ----------------------- | -------------------- | ------------------------------------------------------- | --------------- | --------------- |
| Sum                     | `sum`                | Sum of all elements in set                              | {4, 2, 1}, 7    |                 |
| Longest Sequence Length | `longest_seq_length` | Length of the longest monotonically increasing sequence | {2, 1, 1, 4}, 3 |                 |