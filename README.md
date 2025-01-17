<div align="center">

<img src="images/pointblank_logo.svg" alt="Pointblank logo" width="350px"/>

_Find out if your data is what you think it is._

[![Python Versions](https://img.shields.io/pypi/pyversions/pointblank.svg)](https://pypi.python.org/pypi/pointblank)
[![PyPI](https://img.shields.io/pypi/v/pointblank)](https://pypi.org/project/pointblank/#history)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pointblank)](https://pypistats.org/packages/pointblank)
[![License](https://img.shields.io/github/license/rich-iannone/pointblank)](https://img.shields.io/github/license/rich-iannone/pointblank)

[![CI Build](https://github.com/rich-iannone/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/rich-iannone/pointblank/actions/workflows/ci-tests.yaml)
[![Codecov branch](https://img.shields.io/codecov/c/github/rich-iannone/pointblank/main.svg)](https://codecov.io/gh/rich-iannone/pointblank)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://rich-iannone.github.io/pointblank/)

[![Contributors](https://img.shields.io/github/contributors/rich-iannone/pointblank)](https://github.com/rich-iannone/pointblank/graphs/contributors)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

Pointblank is a table validation and testing library for Python. It helps you ensure that your
tabular data meets certain expectations and constraints and it presents the results in a beautiful,
and useful, validation report table.

## Getting Started

Let's take a Polars DataFrame and validate it against a set of constraints. We do that by using the
`Validate` class along with adding validation steps:

```python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table")) # Use Validate() to start
    .col_vals_gt(columns="d", value=100)       # STEP 1 |
    .col_vals_le(columns="c", value=5)         # STEP 2 | <-- Build up a validation plan
    .col_exists(columns=["date", "date_time"]) # STEP 3 |
    .interrogate() # This will execute all validation steps and collect intel
)

validation
```

<img src="images/pointblank-tabular-report.png" alt="Validation Report">

The rows in the validation table correspond to each of the validation steps. One of the key concepts
is that validation steps can be broken down into atomic test cases (test units) and each of these
test units is given either of pass/fail status based on the validation constraints. You'll see these
tallied up in the reporting table (in the `UNITS`, `PASS`, and `FAIL` columns).

The tabular reporting view is just one way to see the results. You can also obtain fine-grained
results of the interrogation as JSON output or through methods that get key metrics. You can also
utilize the validation results to perform filtering of the input table based on row-level pass/fail
status (via the `get_sundered_data()` method).

On the input side, we can use the following types of tables:

- Polars DataFrame
- Pandas DataFrame
- DuckDB table
- MySQL table
- PostgreSQL table
- SQLite table
- Parquet

To make this all work seamlessly, we use [Narwhals](https://github.com/narwhals-dev/narwhals) to
work with Polars and Pandas DataFrames. We also integrate with
[Ibis](https://github.com/ibis-project/ibis) to enable the use of DuckDB, MySQL, PostgreSQL, SQLite,
Parquet, and more! In doing all of this, we can provide an ergonomic and consistent API for
validating tabular data from various sources.

## Features

Here's a short list of what we think makes Pointblank a great tool for data validation:

- **Declarative Syntax**: Define your data validation rules simply, using a declarative syntax
- **Flexible**: We support tables from Polars, Pandas, Duckdb, MySQL, PostgreSQL, SQLite, and Parquet
- **Beautiful Reports**: Generate beautiful HTML table reports on how the data validation went down
- **Functional Output**: Extract the specific data validation outputs you need for further processing
- **Data Testing**: Write tests for your data and use them in your notebooks or testing framework
- **Easy to Use**: Get started quickly with a simple API and super clear documentation
- **Powerful**: You can develop complex data validation rules with flexible options for customization

There's a lot of [interesting examples](https://rich-iannone.github.io/pointblank/examples/) you can
check out in the documentation website. If you have any questions or would like to request a
feature, you are always welcome to [create an issue](https://github.com/rich-iannone/pointblank/issues)
or [start a discussion](https://github.com/rich-iannone/pointblank/discussions)!

## Installation

You can install Pointblank using pip:

```bash
pip install pointblank
```

If you encounter a bug, have usage questions, or want to share ideas to make this package better,
please feel free to file an [issue](https://github.com/rich-iannone/pointblank/issues).

## Code of Conduct

Please note that the pointblank project is released with a
[contributor code of conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
<br>By participating in this project you agree to abide by its terms.

## Contributing to Pointblank

There are many ways to contribute to the ongoing development of Pointblank. Some contributions can
be simple (like fixing typos, improving documentation, filing issues for feature requests or
problems, etc.) and others might take more time and care (like answering questions and submitting
PRs with code changes). Just know that anything you can do to help would be very much appreciated!

Please read over the
[contributing guidelines](https://github.com/rich-iannone/pointblank/blob/main/CONTRIBUTING.md) for
information on how to get started.

## 📄 License

Pointblank is licensed under the MIT license.

## 🏛️ Governance

This project is primarily maintained by
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Other authors may occasionally
assist with some of these duties.
