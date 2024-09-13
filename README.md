<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines-core Logo" width=500></img>

*Benchmark structured generation libraries.*
</div>

Benchmark suite for the following structured generation libraries:

- [Outlines](https://github.com/outlines-dev/outlines)
- [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)


## Motivation

Discussions around the performance of different structured generation methods tend to revolve around misconceptions. This repository aims at grounding the debate by offering a benchmarking suite for the different implementations. The benchmarking suite is public, and we accept pull requests.

Different methods make different trade-offs, and it is important to know when a method is faster than another. We will highlight differences, ideally using minimum pathological examples.


## Explanations

We do not use models to run the benchmarks, as it would lead to increased runtime, more complex code, and unpredictable generation lengths. We instead take a string in the language of the regular expressions / JSON Schemas, tokenize it and iterate over it pretending these were generated tokens.

### Outlines

If you look at the [benchmarking suite for Outlines](https://github.com/outlines-dev/benchmarks/blob/main/src/outlines.py) you will notice that we execute:

``` python
Regexguide("a", tokenizer)
```

in the initialization phase of the benchmark. This serves two purposes:

1. JIT-compile the functions decorated with `@numba.njit`;
2. Convert vocabulary strings to Numba types.

This only ever needs to be done once, possibly while loading the model, and could be made to disappear using Ahead Of Time compilation. In this benchmarking suite we thus measure:

1. The time it takes to compile the index corresponding to a regular expression;
2. The time it takes to look for valid tokens when generating text.
