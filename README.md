<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines-core Logo" width=500></img>

*Benchmark structured generation libraries.*
</div>

Benchmark suite for the following structured generation libraries:

- [outlines-core](https://github.com/dottxt-ai/outlines-core)
- [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)


## Motivation

Discussions around the performance of different structured generation methods tend to revolve around misconceptions. This repository aims at grounding the debate by offering a benchmarking suite for the different implementations. The benchmarking suite is public, and we accept pull requests.

Different methods make different trade-offs, and it is important to know when a method is faster than another. We will highlight differences, ideally using minimum pathological examples.


## How benchmarks are run

We do not use models to run the benchmarks, as it would lead to increased runtime, more complex code, and unpredictable generation lengths. We instead take a string in the language of the regular expressions / JSON Schemas, tokenize it and iterate over it pretending these were generated tokens.
