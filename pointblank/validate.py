from __future__ import annotations

from importlib_resources import files

import base64
import commonmark
import datetime
import inspect
import json
import re

from dataclasses import dataclass
from typing import Callable, Literal, Any
from zipfile import ZipFile

import narwhals as nw
from narwhals.typing import FrameT
from great_tables import GT, html, loc, style, google_font, from_column, vals

from pointblank._constants import (
    ASSERTION_TYPE_METHOD_MAP,
    COMPATIBLE_DTYPES,
    METHOD_CATEGORY_MAP,
    IBIS_BACKENDS,
    ROW_BASED_VALIDATION_TYPES,
    VALIDATION_REPORT_FIELDS,
    TABLE_TYPE_STYLES,
    SVG_ICONS_FOR_ASSERTION_TYPES,
    SVG_ICONS_FOR_TBL_STATUS,
)
from pointblank._interrogation import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
    ColValsRegex,
    ColExistsHasType,
    NumberOfTestUnits,
)
from pointblank.thresholds import (
    Thresholds,
    _normalize_thresholds_creation,
    _convert_abs_count_to_fraction,
)
from pointblank._utils import _get_fn_name, _check_invalid_fields
from pointblank._utils_check_args import (
    _check_column,
    _check_value_float_int,
    _check_set_types,
    _check_pre,
    _check_thresholds,
    _check_boolean_input,
)

__all__ = ["Validate", "load_dataset", "col"]


def load_dataset(
    dataset: Literal["small_table", "game_revenue"] = "small_table",
    tbl_type: Literal["polars", "pandas", "duckdb"] = "polars",
) -> FrameT | Any:
    """
    Load a dataset hosted in the library as specified DataFrame type.

    Parameters
    ----------
    dataset
        The name of the dataset to load. Current options are `"small_table"` and `"game_revenue"`.
    tbl_type
        The type of DataFrame to generate from the dataset. The named options are `"polars"`,
        `"pandas"`, and `"duckdb"`.

    Returns
    -------
    FrameT | Any
        The dataset for the `Validate` object. This could be a Polars DataFrame, a Pandas DataFrame,
        or a DuckDB table as an Ibis table.

    Examples
    --------
    Load the `small_table` dataset as a Polars DataFrame by calling `load_dataset()` with its
    defaults:

    ```{python}
    import pointblank as pb

    small_table = pb.load_dataset()

    small_table
    ```

    The `game_revenue` dataset can be loaded as a Pandas DataFrame by specifying the dataset name
    and setting `tbl_type="pandas"`:

    ```{python}
    import pointblank as pb

    game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="pandas")

    game_revenue
    ```
    """

    # Raise an error if the dataset is from the list of provided datasets
    if dataset not in ["small_table", "game_revenue"]:
        raise ValueError(
            f"The dataset name `{dataset}` is not valid. Choose one of the following:\n"
            "- `small_table`\n"
            "- `game_revenue`"
        )

    # Raise an error if the `tbl_type=` value is not of the supported types
    if tbl_type not in ["polars", "pandas", "duckdb"]:
        raise ValueError(
            f"The DataFrame type `{tbl_type}` is not valid. Choose one of the following:\n"
            "- `polars`\n"
            "- `pandas`\n"
            "- `duckdb`"
        )

    data_path = files("pointblank.data") / f"{dataset}.zip"

    if tbl_type == "polars":

        if not _is_lib_present(lib_name="polars"):
            raise ImportError(
                "The Polars library is not installed but is required when specifying "
                '`tbl_type="polars".'
            )

        import polars as pl

        dataset = pl.read_csv(ZipFile(data_path).read(f"{dataset}.csv"), try_parse_dates=True)

    if tbl_type == "pandas":

        if not _is_lib_present(lib_name="pandas"):
            raise ImportError(
                "The Pandas library is not installed but is required when specifying "
                '`tbl_type="pandas".'
            )

        import pandas as pd

        parse_date_columns = {
            "small_table": ["date_time", "date"],
            "game_revenue": ["session_start", "time", "start_day"],
        }

        dataset = pd.read_csv(data_path, parse_dates=parse_date_columns[dataset])

    if tbl_type == "duckdb":  # pragma: no cover

        if not _is_lib_present(lib_name="ibis"):
            raise ImportError(
                "The Ibis library is not installed but is required when specifying "
                '`tbl_type="duckdb".'
            )

        import ibis

        data_path = files("pointblank.data") / f"{dataset}-duckdb.zip"

        # Unzip the DuckDB dataset to a temporary directory
        with ZipFile(data_path, "r") as z:

            z.extractall(path="datasets")

            data_path = f"datasets/{dataset}.ddb"

            dataset = ibis.connect(f"duckdb://{data_path}").table(dataset)

    return dataset


@dataclass
class _ValidationInfo:
    """
    Information about a validation to be performed on a table and the results of the interrogation.

    Attributes
    ----------
    i
        The validation step number.
    i_o
        The original validation step number (if a step creates multiple steps). Unused.
    step_id
        The ID of the step (if a step creates multiple steps). Unused.
    sha1
        The SHA-1 hash of the step. Unused.
    assertion_type
        The type of assertion. This is the method name of the validation (e.g., `"col_vals_gt"`).
    column
        The column to validate. Currently we don't allow for column expressions (which may map to
        multiple columns).
    values
        The value or values to compare against.
    na_pass
        Whether to pass test units that hold missing values.
    pre
        A pre-processing function or lambda to apply to the data table for the validation step.
    thresholds
        The threshold values for the validation.
    label
        A label for the validation step. Unused.
    brief
        A brief description of the validation step. Unused.
    active
        Whether the validation step is active.
    all_passed
        Upon interrogation, this describes whether all test units passed for a validation step.
    n
        The number of test units for the validation step.
    n_passed
        The number of test units that passed (i.e., passing test units).
    n_failed
        The number of test units that failed (i.e., failing test units).
    f_passed
        The fraction of test units that passed. The calculation is `n_passed / n`.
    f_failed
        The fraction of test units that failed. The calculation is `n_failed / n`.
    warn
        Whether the number of failing test units is beyond the warning threshold.
    stop
        Whether the number of failing test units is beyond the stopping threshold.
    notify
        Whether the number of failing test units is beyond the notification threshold.
    tbl_checked
        The data table in its native format that has been checked for the validation step. It wil
        include a new column called `pb_is_good_` that is a boolean column that indicates whether
        the row passed the validation or not.
    extract
        The extracted rows from the table that failed the validation step.
    time_processed
        The time the validation step was processed. This is in the ISO 8601 format in UTC time.
    proc_duration_s
        The duration of processing for the validation step in seconds.
    """

    # Validation plan
    i: int | None = None
    i_o: int | None = None
    step_id: str | None = None
    sha1: str | None = None
    assertion_type: str | None = None
    column: str | None = None
    values: any | list[any] | tuple | None = None
    inclusive: tuple[bool, bool] | None = None
    na_pass: bool | None = None
    pre: Callable | None = None
    thresholds: Thresholds | None = None
    label: str | None = None
    brief: str | None = None
    active: bool | None = None
    # Interrogation results
    all_passed: bool | None = None
    n: int | None = None
    n_passed: int | None = None
    n_failed: int | None = None
    f_passed: int | None = None
    f_failed: int | None = None
    warn: bool | None = None
    stop: bool | None = None
    notify: bool | None = None
    tbl_checked: FrameT | None = None
    extract: FrameT | None = None
    time_processed: str | None = None
    proc_duration_s: float | None = None


@dataclass
class Validate:
    """
    Workflow for defining a set of validations on a table and interrogating for results.

    The `Validate` class is used for defining a set of validation steps on a table and interrogating
    the table with the *validation plan*. This class is the main entry point for the *data quality
    reporting* workflow. The overall aim of this workflow is to generate comprehensive reporting
    information to assess the level of data quality for a target table.

    We can supply as many validation steps as needed, and having a large number of them should
    increase the validation coverage for a given table. The validation methods (e.g.,
    `col_vals_gt()`, `col_vals_between()`, etc.) translate to discrete validation steps, where each
    step will be sequentially numbered (useful when viewing the reporting data). This process of
    calling validation methods is known as developing a *validation plan*.

    The validation methods, when called, are merely instructions up to the point the concluding
    `interrogate()` method is called. That kicks off the process of acting on the *validation plan*
    by querying the target table getting reporting results for each step. Once the interrogation
    process is complete, we can say that the workflow now has reporting information. We can then
    extract useful information from the reporting data to understand the quality of the table. For
    instance `get_tabular_report()` method which will return a table with the results of the
    interrogation and `get_sundered_data()` allows for the splitting of the table based on passing
    and failing rows.

    Parameters
    ----------
    data
        The table to validate. Can be any of the table types described in the *Supported Input
        Table Types* section.
    tbl_name
        A optional name to assign to the input table object. If no value is provided, a name will
        be generated based on whatever information is available. This table name will be displayed
        in the header area of the HTML report generated by using the `get_tabular_report()` method.
    label
        An optional label for the validation plan. If no value is provided, a label will be
        generated based on the current system date and time. Markdown can be used here to make the
        label more visually appealing (it will appear in the header area of the HTML report).
    thresholds
        Generate threshold failure levels so that all validation steps can report and react
        accordingly when exceeding the set levels. This is to be created using one of several valid
        input schemes: (1) single integer/float denoting absolute number or fraction of failing test
        units for the 'warn' level, (2) a tuple of 1-3 values, (3) a dictionary of 1-3 entries, or a
        `Thresholds` object.

    Returns
    -------
    Validate
        A `Validate` object with the table and validations to be performed.

    Supported Input Table Types
    ---------------------------
    The `data=` parameter can be given any of the following table types:

    - Polars DataFrame (`"polars"`)
    - Pandas DataFrame (`"pandas"`)
    - DuckDB table (`"duckdb"`)*
    - MySQL table (`"mysql"`)*
    - PostgreSQL table (`"postgresql"`)*
    - SQLite table (`"sqlite"`)*
    - Parquet table (`"parquet"`)*

    The table types marked with an asterisk need to be prepared as Ibis tables (with type of
    `ibis.expr.types.relations.Table`). Furthermore, the use of `Validate` with such tables requires
    the Ibis library v9.5.0 and above to be installed. If the input table is a Polars or Pandas
    DataFrame, the Ibis library is not required.

    Examples
    --------
    ## Creating a validation plan and interrogating

    Let's walk through a data quality analysis of an extremely small table. It's actually called
    `small_table` and it's accessible through the `load_dataset()` function.

    ```{python}
    import pointblank as pb

    # Load the small_table dataset
    small_table = pb.load_dataset()

    small_table
    ```

    We ought to think about what's tolerable in terms of data quality so let's designate
    proportional failure thresholds to the **warn**, **stop**, and **notify** states. This can be
    done by using the `Thresholds` class.

    ```{python}
    thresholds = pb.Thresholds(warn_at=0.10, stop_at=0.25, notify_at=0.35)
    ```

    Now, we use the `Validate` class and give it the `thresholds` object (which serves as a default
    for all validation steps but can be overridden). The static thresholds provided in `thresholds`
    will make the reporting a bit more useful. We also need to provide a target table and we'll use
    `small_table` for this.

    ```{python}
    validation = (
        pb.Validate(
            data=small_table,
            tbl_name="small_table",
            label="`Validate` example.",
            thresholds=thresholds
        )
    )
    ```

    Then, as with any `Validate` object, we can add steps to the validation plan by using as many
    validation methods as we want. To conclude the process (and actually query the data table), we
    use the `interrogate()` method.

    ```{python}
    validation = (
        validation
        .col_vals_gt(columns="d", value=100)
        .col_vals_le(columns="c", value=5)
        .col_vals_between(columns="c", left=3, right=10, na_pass=True)
        .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .col_exists(columns=["date", "date_time"])
        .interrogate()
    )
    ```

    The `validation` object can be printed as a reporting table.

    ```{python}
    validation
    ```
    """

    data: FrameT
    tbl_name: str | None = None
    label: str | None = None
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None

    def __post_init__(self):

        # Check input of the `thresholds=` argument
        _check_thresholds(thresholds=self.thresholds)

        # Normalize the thresholds value (if any) to a Thresholds object
        self.thresholds = _normalize_thresholds_creation(self.thresholds)

        # TODO: Add functionality to obtain the column names and types from the table
        self.col_names = None
        self.col_types = None

        self.time_start = None
        self.time_end = None

        self.validation_info = []

    def _repr_html_(self) -> str:

        return self.get_tabular_report()._repr_html_()  # pragma: no cover

    def col_vals_gt(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data greater than a fixed value or data in another column?

        The `col_vals_gt()` validation method checks whether column values in a table are
        *greater than* a specified `value=` (the exact comparison used in this function is
        `col_val > value`). The `value=` can be specified as a single, literal value or as a column
        name given in `col()`. This validation will operate over the number of test units that is
        equal to the number of rows in the table (determined after any `pre=` mutation has been
        applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 7, 6, 5],
                "b": [1, 2, 1, 2, 2, 2],
                "c": [2, 1, 2, 2, 3, 4],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all greater than the value of `4`. We'll
        determine if this validation had any failing test units (there are six test units, one for
        each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=4)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_gt()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_gt()` to check whether the values in column `c`
        are greater than values in column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="c", value=pb.col("b"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 1: `c` is `1` and `b` is `2`.
        - Row 3: `c` is `2` and `b` is `2`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_lt(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data less than a fixed value or data in another column?

        The `col_vals_lt()` validation method checks whether column values in a table are
        *less than* a specified `value=` (the exact comparison used in this function is
        `col_val < value`). The `value=` can be specified as a single, literal value or as a column
        name given in `col()`. This validation will operate over the number of test units that is
        equal to the number of rows in the table (determined after any `pre=` mutation has been
        applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 9, 7, 5],
                "b": [1, 2, 1, 2, 2, 2],
                "c": [2, 1, 1, 4, 3, 4],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all less than the value of `10`. We'll
        determine if this validation had any failing test units (there are six test units, one for
        each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_lt(columns="a", value=10)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_lt()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_lt()` to check whether the values in column `b`
        are less than values in column `c`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_lt(columns="b", value=pb.col("c"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 1: `b` is `2` and `c` is `1`.
        - Row 2: `b` is `1` and `c` is `1`.
        """
        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_eq(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data equal to a fixed value or data in another column?

        The `col_vals_eq()` validation method checks whether column values in a table are
        *equal to* a specified `value=` (the exact comparison used in this function is
        `col_val == value`). The `value=` can be specified as a single, literal value or as a column
        name given in `col()`. This validation will operate over the number of test units that is
        equal to the number of rows in the table (determined after any `pre=` mutation has been
        applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 5, 5, 5, 5, 5],
                "b": [5, 5, 5, 6, 5, 4],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all equal to the value of `5`. We'll determine
        if this validation had any failing test units (there are six test units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_eq(columns="a", value=5)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_eq()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_eq()` to check whether the values in column `a`
        are equal to the values in column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_eq(columns="a", value=pb.col("b"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 3: `a` is `5` and `b` is `6`.
        - Row 5: `a` is `5` and `b` is `4`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_ne(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data not equal to a fixed value or data in another column?

        The `col_vals_ne()` validation method checks whether column values in a table are
        *not equal to* a specified `value=` (the exact comparison used in this function is
        `col_val != value`). The `value=` can be specified as a single, literal value or as a column
        name given in `col()`. This validation will operate over the number of test units that is
        equal to the number of rows in the table (determined after any `pre=` mutation has been
        applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 5, 5, 5, 5, 5],
                "b": [5, 6, 3, 6, 5, 8],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are not equal to the value of `3`. We'll determine
        if this validation had any failing test units (there are six test units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_ne(columns="a", value=3)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_ne()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_ne()` to check whether the values in column `a`
        aren't equal to the values in column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_ne(columns="a", value=pb.col("b"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are in rows
        0 and 4, where `a` is `5` and `b` is `5` in both cases (i.e., they are equal to each other).
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_ge(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data greater than or equal to a fixed value or data in another column?

        The `col_vals_ge()` validation method checks whether column values in a table are
        *greater than or equal to* a specified `value=` (the exact comparison used in this function
        is `col_val >= value`). The `value=` can be specified as a single, literal value or as a
        column name given in `col()`. This validation will operate over the number of test units
        that is equal to the number of rows in the table (determined after any `pre=` mutation has
        been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 9, 7, 5],
                "b": [5, 3, 1, 8, 2, 3],
                "c": [2, 3, 1, 4, 3, 4],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all greater than or equal to the value of `5`.
        We'll determine if this validation had any failing test units (there are six test units, one
        for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_ge(columns="a", value=5)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_ge()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_ge()` to check whether the values in column `b`
        are greater than values in column `c`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_ge(columns="b", value=pb.col("c"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 0: `b` is `2` and `c` is `3`.
        - Row 4: `b` is `3` and `c` is `4`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_le(
        self,
        columns: str | list[str],
        value: float | int,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Are column data less than or equal to a fixed value or data in another column?

        The `col_vals_le()` validation method checks whether column values in a table are
        *less than or equal to* a specified `value=` (the exact comparison used in this function is
        `col_val <= value`). The `value=` can be specified as a single, literal value or as a column
        name given in `col()`. This validation will operate over the number of test units that is
        equal to the number of rows in the table (determined after any `pre=` mutation has been
        applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        value
            The value to compare against. This can be a single numeric value or a column name given
            in `col()`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 9, 7, 5],
                "b": [1, 3, 1, 5, 2, 5],
                "c": [2, 1, 1, 4, 3, 4],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all less than or equal to the value of `9`.
        We'll determine if this validation had any failing test units (there are six test units, one
        for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_le(columns="a", value=9)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_le()`. All test units passed, and there are no failing test units.

        Aside from checking a column against a literal value, we can also use a column name in the
        `value=` argument (with the helper function `col()`) to perform a column-column comparison.
        For the next example, we'll use `col_vals_le()` to check whether the values in column `c`
        are less than values in column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_le(columns="c", value=pb.col("b"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 0: `c` is `2` and `b` is `1`.
        - Row 4: `c` is `3` and `b` is `2`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=value)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_between(
        self,
        columns: str | list[str],
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Do column data lie between two specified values or data in other columns?

        The `col_vals_between()` validation method checks whether column values in a table fall
        within a range. The range is specified with three arguments: `left=`, `right=`, and
        `inclusive=`. The `left=` and `right=` values specify the lower and upper bounds. These
        bounds can be specified as literal values or as column names provided within `col()`. The
        validation will operate over the number of test units that is equal to the number of rows in
        the table (determined after any `pre=` mutation has been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        left
            The lower bound of the range. Can be a single numeric value or a column name given in
            `col()`.
        right
            The upper bound of the range. Can be a single numeric value or a column name given in
            `col()`.
        inclusive
            A tuple of two boolean values indicating whether the comparison should be inclusive. The
            position of the boolean values correspond to the `left=` and `right=` values,
            respectively. By default, both values are `True`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [2, 3, 2, 4, 3, 4],
                "b": [5, 6, 1, 6, 8, 5],
                "c": [9, 8, 8, 7, 7, 8],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all between the fixed boundary values of `1`
        and `5`. We'll determine if this validation had any failing test units (there are six test
        units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="a", left=1, right=5)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_between()`. All test units passed, and there are no failing test units.

        Aside from checking a column against two literal values representing the lower and upper
        bounds, we can also provide column names to the `left=` and/or `right=` arguments (by using
        the helper function `col()`). In this way, we can perform three additional comparison types:

        1. `left=column`, `right=column`
        2. `left=literal`, `right=column`
        3. `left=column`, `right=literal`

        For the next example, we'll use `col_vals_between()` to check whether the values in column
        `b` are between than corresponding values in columns `a` (lower bound) and `c` (upper
        bound).

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="b", left=pb.col("a"), right=pb.col("c"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 2: `b` is `1` but the bounds are `2` (`a`) and `8` (`c`).
        - Row 4: `b` is `8` but the bounds are `3` (`a`) and `7` (`c`).
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=left)
        _check_value_float_int(value=right)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        value = (left, right)

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                inclusive=inclusive,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_outside(
        self,
        columns: str | list[str],
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Do column data lie outside of two specified values or data in other columns?

        The `col_vals_between()` validation method checks whether column values in a table *do not*
        fall within a certain range. The range is specified with three arguments: `left=`, `right=`,
        and `inclusive=`. The `left=` and `right=` values specify the lower and upper bounds. These
        bounds can be specified as literal values or as column names provided within `col()`. The
        validation will operate over the number of test units that is equal to the number of rows in
        the table (determined after any `pre=` mutation has been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        left
            The lower bound of the range. Can be a single numeric value or a column name given in
            `col()`.
        right
            The upper bound of the range. Can be a single numeric value or a column name given in
            `col()`.
        inclusive
            A tuple of two boolean values indicating whether the comparison should be inclusive. The
            position of the boolean values correspond to the `left=` and `right=` values,
            respectively. By default, both values are `True`.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`,
        `b`, and `c`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 7, 5, 5],
                "b": [2, 3, 6, 4, 3, 6],
                "c": [9, 8, 8, 9, 9, 7],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all outside the fixed boundary values of `1`
        and `4`. We'll determine if this validation had any failing test units (there are six test
        units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_outside(columns="a", left=1, right=4)
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_outside()`. All test units passed, and there are no failing test units.

        Aside from checking a column against two literal values representing the lower and upper
        bounds, we can also provide column names to the `left=` and/or `right=` arguments (by using
        the helper function `col()`). In this way, we can perform three additional comparison types:

        1. `left=column`, `right=column`
        2. `left=literal`, `right=column`
        3. `left=column`, `right=literal`

        For the next example, we'll use `col_vals_outside()` to check whether the values in column
        `b` are outside of the range formed by the corresponding values in columns `a` (lower bound)
        and `c` (upper bound).

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_outside(columns="b", left=pb.col("a"), right=pb.col("c"))
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are:

        - Row 2: `b` is `6` and the bounds are `5` (`a`) and `8` (`c`).
        - Row 5: `b` is `6` and the bounds are `5` (`a`) and `7` (`c`).
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_value_float_int(value=left)
        _check_value_float_int(value=right)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        value = (left, right)

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=value,
                inclusive=inclusive,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_in_set(
        self,
        columns: str | list[str],
        set: list[float | int],
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether column values are in a set of values.

        The `col_vals_in_set()` validation method checks whether column values in a table are part
        of a specified `set=` of values. This validation will operate over the number of test units
        that is equal to the number of rows in the table (determined after any `pre=` mutation has
        been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        set
            A list of values to compare against.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [5, 2, 4, 6, 2, 5],
                "b": [5, 8, 2, 6, 5, 1],
            }
        )

        tbl
        ```

        Let's validate that values in column `a` are all in the set of `[2, 3, 4, 5, 6]`. We'll
        determine if this validation had any failing test units (there are six test units, one for
        each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_in_set(columns="a", set=[2, 3, 4, 5, 6])
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_in_set()`. All test units passed, and there are no failing test units.

        Now, let's use that same set of values for a validation on column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_in_set(columns="b", set=[2, 3, 4, 5, 6])
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are for the
        column `b` values of `8` and `1`, which are not in the set of `[2, 3, 4, 5, 6]`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_set_types(set=set)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=set,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_not_in_set(
        self,
        columns: str | list[str],
        set: list[float | int],
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether column values are not in a set of values.

        The `col_vals_not_in_set()` validation method checks whether column values in a table are
        *not* part of a specified `set=` of values. This validation will operate over the number of
        test units that is equal to the number of rows in the table (determined after any `pre=`
        mutation has been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        set
            A list of values to compare against.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [7, 8, 1, 9, 1, 7],
                "b": [1, 8, 2, 6, 9, 1],
            }
        )

        tbl
        ```

        Let's validate that none of the values in column `a` are in the set of `[2, 3, 4, 5, 6]`.
        We'll determine if this validation had any failing test units (there are six test units, one
        for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_not_in_set(columns="a", set=[2, 3, 4, 5, 6])
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_not_in_set()`. All test units passed, and there are no failing test
        units.

        Now, let's use that same set of values for a validation on column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_not_in_set(columns="b", set=[2, 3, 4, 5, 6])
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are for the
        column `b` values of `2` and `6`, both of which are in the set of `[2, 3, 4, 5, 6]`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_set_types(set=set)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=set,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_null(
        self,
        columns: str | list[str],
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether values in a column are NULL.

        The `col_vals_null()` validation method checks whether column values in a table are NULL.
        This validation will operate over the number of test units that is equal to the number
        of rows in the table.

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [None, None, None, None],
                "b": [None, 2, None, 9],
            }
        ).with_columns(pl.col("a").cast(pl.Int64))

        tbl
        ```

        Let's validate that values in column `a` are all Null values. We'll determine if this
        validation had any failing test units (there are four test units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_null(columns="a")
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_null()`. All test units passed, and there are no failing test units.

        Now, let's use that same set of values for a validation on column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_null(columns="b")
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are for the
        two non-Null values in column `b`.
        """
        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_not_null(
        self,
        columns: str | list[str],
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether values in a column are not NULL.

        The `col_vals_not_null()` validation method checks whether column values in a table are not
        NULL. This validation will operate over the number of test units that is equal to the number
        of rows in the table.

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two numeric columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": [4, 7, 2, 8],
                "b": [5, None, 1, None],
            }
        )

        tbl
        ```

        Let's validate that none of the values in column `a` are Null values. We'll determine if
        this validation had any failing test units (there are four test units, one for each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_not_null(columns="a")
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_not_null()`. All test units passed, and there are no failing test units.

        Now, let's use that same set of values for a validation on column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_not_null(columns="b")
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are for the
        two Null values in column `b`.
        """
        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_vals_regex(
        self,
        columns: str | list[str],
        pattern: str,
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether column values match a regular expression pattern.

        The `col_vals_regex()` validation method checks whether column values in a table
        correspond to a `pattern=` matching expression. This validation will operate over the number
        of test units that is equal to the number of rows in the table (determined after any `pre=`
        mutation has been applied).

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        pattern
            A regular expression pattern to compare against.
        na_pass
            Should any encountered None, NA, or Null values be considered as passing test units? By
            default, this is `False`. Set to `True` to pass test units with missing values.
        pre
            A pre-processing function or lambda to apply to the data table for the validation step.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with two string columns (`a` and
        `b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": ["rb-0343", "ra-0232", "ry-0954", "rc-1343"],
                "b": ["ra-0628", "ra-583", "rya-0826", "rb-0735"],
            }
        )

        tbl
        ```

        Let's validate that all of the values in column `a` match a particular regex pattern. We'll
        determine if this validation had any failing test units (there are four test units, one for
        each row).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_vals_regex(columns="a", pattern=r"r[a-z]-\d{4}")
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows the single entry that corresponds to the validation step created
        by using `col_vals_regex()`. All test units passed, and there are no failing test units.

        Now, let's use the same regex for a validation on column `b`.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_vals_regex(columns="b", pattern=r"r[a-z]-\d{4}")
            .interrogate()
        )

        validation
        ```

        The validation table reports two failing test units. The specific failing cases are for the
        string values of rows 1 and 2 in column `b`.
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_pre(pre=pre)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=pattern,
                na_pass=na_pass,
                pre=pre,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def col_exists(
        self,
        columns: str | list[str],
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Validate whether one or more columns exist in the table.

        The `col_exists()` method checks whether one or more columns exist in the target table. The
        only requirement is specification of the column names. Each validation step or expectation
        will operate over a single test unit, which is whether the column exists or not.

        Parameters
        ----------
        columns
            A single column or a list of columns to validate. If multiple columns are supplied,
            there will be a separate validation step generated for each column.
        thresholds
            Failure threshold levels so that the validation step can react accordingly when
            exceeding the set levels for different states (`warn`, `stop`, and `notify`). This can
            be created simply as an integer or float denoting the absolute number or fraction of
            failing test units for the 'warn' level. Otherwise, you can use a tuple of 1-3 values,
            a dictionary of 1-3 entries, or a Thresholds object.
        active
            A boolean value indicating whether the validation step should be active. Using `False`
            will make the validation step inactive (still reporting its presence and keeping indexes
            for the steps unchanged).

        Returns
        -------
        Validate
            The `Validate` object with the added validation step.

        Examples
        --------
        For the examples here, we'll use a simple Polars DataFrame with a string columns (`a`) and a
        numeric column (`b`). The table is shown below:

        ```{python}
        import polars as pl

        tbl = pl.DataFrame(
            {
                "a": ["apple", "banana", "cherry", "date"],
                "b": [1, 6, 3, 5],
            }
        )

        tbl
        ```

        Let's validate that the columns `a` and `b` actually exist in the table. We'll determine if
        this validation had any failing test units (each validation will have a single test unit).

        ```{python}
        import pointblank as pb

        validation = (
            pb.Validate(data=tbl)
            .col_exists(columns=["a", "b"])
            .interrogate()
        )

        validation
        ```

        Printing the `validation` object shows the validation table in an HTML viewing environment.
        The validation table shows two entries (one check per column) generated by the
        `col_exists()` validation step. Both steps passed since both columns provided in `columns=`
        are present in the table.

        Now, let's check for the existence of a different set of columns.

        ```{python}
        validation = (
            pb.Validate(data=tbl)
            .col_exists(columns=["b", "c"])
            .interrogate()
        )

        validation
        ```

        The validation table reports one passing validation step (the check for column `b`) and one
        failing validation step (the check for column `c`, which doesn't exist).
        """

        assertion_type = _get_fn_name()

        _check_column(column=columns)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        if isinstance(columns, str):
            columns = [columns]

        # Determine threshold to use (global or local) and normalize a local `thresholds=` value
        thresholds = (
            self.thresholds if thresholds is None else _normalize_thresholds_creation(thresholds)
        )

        for column in columns:

            val_info = _ValidationInfo(
                assertion_type=assertion_type,
                column=column,
                values=None,
                thresholds=thresholds,
                active=active,
            )

            self._add_validation(validation_info=val_info)

        return self

    def interrogate(
        self,
        collect_extracts: bool = True,
        collect_tbl_checked: bool = True,
        get_first_n: int | None = None,
        sample_n: int | None = None,
        sample_frac: int | float | None = None,
        sample_limit: int = 5000,
    ) -> Validate:
        """
        Evaluate each validation against the table and store the results.

        When a validation plan has been set with a series of validation steps, the interrogation
        process through `interrogate()` should then be invoked. Interrogation will evaluate each
        validation step against the table and store the results. The interrogation process is
        non-destructive; the original table is not altered (but a copy is made for each validation).

        After that, the `Validate` object will have gathered information, and we can use methods
        like `get_tabular_report()`, `all_passed()` and many more to understand how the table
        performed against the validation plan.

        Parameters
        ----------
        collect_extracts
            An option to collect rows of the input table that didn't pass a particular validation
            step. The default is `True` and further options (i.e., `get_first_n=`, `sample_*=`)
            allow for fine control of how these rows are collected.
        collect_tbl_checked
            The processed data frames produced by executing the validation steps is collected and
            stored in the `Validate` object if `collect_tbl_checked=True`. This information is
            necessary for some methods (e.g., `get_sundered_data()`), but it potentially makes the
            object grow to a large size. To opt out of attaching this data, set this argument to
            `False`.
        get_first_n
            If the option to collect rows where test units is chosen, there is the option here to
            collect the first `n` rows. Supply an integer number of rows to extract from the top of
            subset table containing non-passing rows (the ordering of data from the original table
            is retained).
        sample_n
            If the option to collect non-passing rows is chosen, this option allows for the
            sampling of `n` rows. Supply an integer number of rows to sample from the subset table.
            If `n` happens to be greater than the number of non-passing rows, then all such rows
            will be returned.
        sample_frac
            If the option to collect non-passing rows is chosen, this option allows for the sampling
            of a fraction of those rows. Provide a number in the range of `0` and `1`. The number of
            rows to return could be very large, however, the `sample_limit=` option will apply a
            hard limit to the returned rows.
        sample_limit
            A value that limits the possible number of rows returned when sampling non-passing rows
            using the `sample_frac=` option.

        Returns
        -------
        Validate
            The `Validate` object with the results of the interrogation.
        """

        # Raise if `get_first_n` and either or `sample_n` or `sample_frac` arguments are provided
        if get_first_n is not None and (sample_n is not None or sample_frac is not None):
            raise ValueError(
                "The `get_first_n=` argument cannot be provided with the `sample_n=` or "
                "`sample_frac=` arguments."
            )

        # Raise if the `sample_n` and `sample_frac` arguments are both provided
        if sample_n is not None and sample_frac is not None:
            raise ValueError(
                "The `sample_n=` and `sample_frac=` arguments cannot both be provided."
            )

        data_tbl = self.data

        # Determine if the table is a DataFrame or a DB table
        tbl_type = _get_tbl_type(data=data_tbl)

        self.time_start = datetime.datetime.now(datetime.timezone.utc)

        for validation in self.validation_info:

            start_time = datetime.datetime.now(datetime.timezone.utc)

            # Skip the validation step if it is not active but still record the time of processing
            if not validation.active:
                end_time = datetime.datetime.now(datetime.timezone.utc)
                validation.proc_duration_s = (end_time - start_time).total_seconds()
                validation.time_processed = end_time.isoformat(timespec="milliseconds")
                continue

            # Make a copy of the table for this step
            data_tbl_step = data_tbl

            # ------------------------------------------------
            # Pre-processing stage
            # ------------------------------------------------

            # Determine whether any pre-processing functions are to be applied to the table
            if validation.pre is not None:

                # Read the text of the pre-processing function
                pre_text = _pre_processing_funcs_to_str(validation.pre)

                # Determine if the pre-processing function is a lambda function; return a boolean
                is_lambda = re.match(r"^lambda", pre_text) is not None

                # If the pre-processing function is a lambda function, then check if there is
                # a keyword argument called `dfn` in the lamda signature; if so, that's a cue
                # to use a Narwhalified version of the table
                if is_lambda:

                    # Get the signature of the lambda function
                    sig = inspect.signature(validation.pre)

                    # Check if the lambda function has a keyword argument called `dfn`
                    if "dfn" in sig.parameters:

                        # Convert the table to a Narwhals DataFrame
                        data_tbl_step = nw.from_native(data_tbl_step)

                        # Apply the pre-processing function to the table
                        data_tbl_step = validation.pre(dfn=data_tbl_step)

                        # Convert the table back to its original format
                        data_tbl_step = nw.to_native(data_tbl_step)

                    else:
                        # Apply the pre-processing function to the table
                        data_tbl_step = validation.pre(data_tbl_step)

                # If the pre-processing function is a function, apply it to the table
                elif isinstance(validation.pre, Callable):

                    data_tbl_step = validation.pre(data_tbl_step)

            assertion_type = validation.assertion_type
            column = validation.column
            value = validation.values
            inclusive = validation.inclusive
            na_pass = validation.na_pass
            threshold = validation.thresholds

            assertion_method = ASSERTION_TYPE_METHOD_MAP[assertion_type]
            assertion_category = METHOD_CATEGORY_MAP[assertion_method]
            compatible_dtypes = COMPATIBLE_DTYPES.get(assertion_method, [])

            validation.n = NumberOfTestUnits(df=data_tbl_step, column=column).get_test_units(
                tbl_type=tbl_type
            )

            if tbl_type not in IBIS_BACKENDS:
                tbl_type = "local"

            if assertion_category == "COMPARE_ONE":

                results_tbl = ColValsCompareOne(
                    data_tbl=data_tbl_step,
                    column=column,
                    value=value,
                    na_pass=na_pass,
                    threshold=threshold,
                    assertion_method=assertion_method,
                    allowed_types=compatible_dtypes,
                    tbl_type=tbl_type,
                ).get_test_results()

            if assertion_category == "COMPARE_TWO":

                results_tbl = ColValsCompareTwo(
                    data_tbl=data_tbl_step,
                    column=column,
                    value1=value[0],
                    value2=value[1],
                    inclusive=inclusive,
                    na_pass=na_pass,
                    threshold=threshold,
                    assertion_method=assertion_method,
                    allowed_types=compatible_dtypes,
                    tbl_type=tbl_type,
                ).get_test_results()

            if assertion_category == "COMPARE_SET":

                inside = True if assertion_method == "in_set" else False

                results_tbl = ColValsCompareSet(
                    data_tbl=data_tbl_step,
                    column=column,
                    values=value,
                    threshold=threshold,
                    inside=inside,
                    allowed_types=compatible_dtypes,
                    tbl_type=tbl_type,
                ).get_test_results()

            if assertion_category == "COMPARE_REGEX":

                results_tbl = ColValsRegex(
                    data_tbl=data_tbl_step,
                    column=column,
                    pattern=value,
                    na_pass=na_pass,
                    threshold=threshold,
                    allowed_types=compatible_dtypes,
                    tbl_type=tbl_type,
                ).get_test_results()

            if assertion_category == "COL_EXISTS_HAS_TYPE":

                result_bool = ColExistsHasType(
                    data_tbl=data_tbl_step,
                    column=column,
                    threshold=threshold,
                    assertion_method="exists",
                    tbl_type=tbl_type,
                ).get_test_results()

                validation.all_passed = result_bool
                validation.n = 1
                validation.n_passed = result_bool
                validation.n_failed = 1 - result_bool

                results_tbl = None

            if assertion_category != "COL_EXISTS_HAS_TYPE":

                # Extract the `pb_is_good_` column from the table as a results list
                if tbl_type in IBIS_BACKENDS:
                    results_list = (
                        results_tbl.select("pb_is_good_").to_pandas()["pb_is_good_"].to_list()
                    )

                else:
                    results_list = nw.from_native(results_tbl)["pb_is_good_"].to_list()

                validation.all_passed = all(results_list)
                validation.n = len(results_list)
                validation.n_passed = results_list.count(True)
                validation.n_failed = results_list.count(False)

            # Calculate fractions of passing and failing test units
            # - `f_passed` is the fraction of test units that passed
            # - `f_failed` is the fraction of test units that failed
            for attr in ["passed", "failed"]:
                setattr(
                    validation,
                    f"f_{attr}",
                    _convert_abs_count_to_fraction(
                        value=getattr(validation, f"n_{attr}"), test_units=validation.n
                    ),
                )

            # Determine if the number of failing test units is beyond the threshold value
            # for each of the severity levels
            # - `warn` is the threshold for a warning
            # - `stop` is the threshold for stopping
            # - `notify` is the threshold for notifying
            for level in ["warn", "stop", "notify"]:
                setattr(
                    validation,
                    level,
                    threshold._threshold_result(
                        fraction_failing=validation.f_failed, test_units=validation.n, level=level
                    ),
                )

            # Include the results table that has a new column called `pb_is_good_`; that
            # is a boolean column that indicates whether the row passed the validation or not
            if collect_tbl_checked and results_tbl is not None:
                validation.tbl_checked = results_tbl

            # If this is a row-based validation step, then extract the rows that failed
            # TODO: Add support for extraction of rows for Ibis backends
            if (
                collect_extracts
                and assertion_type in ROW_BASED_VALIDATION_TYPES
                and tbl_type not in IBIS_BACKENDS
            ):

                validation_extract_nw = (
                    nw.from_native(results_tbl)
                    .filter(nw.col("pb_is_good_") == False)  # noqa
                    .drop("pb_is_good_")
                )

                # Apply any sampling or limiting to the number of rows to extract
                if get_first_n is not None:
                    validation_extract_nw = validation_extract_nw.head(get_first_n)
                elif sample_n is not None:
                    validation_extract_nw = validation_extract_nw.sample(n=sample_n)
                elif sample_frac is not None:
                    validation_extract_nw = validation_extract_nw.sample(fraction=sample_frac)

                    # Ensure a limit is set on the number of rows to extract
                    if len(validation_extract_nw) > sample_limit:
                        validation_extract_nw = validation_extract_nw.head(sample_limit)

                validation.extract = nw.to_native(validation_extract_nw)

            # Get the end time for this step
            end_time = datetime.datetime.now(datetime.timezone.utc)

            # Calculate the duration of processing for this step
            validation.proc_duration_s = (end_time - start_time).total_seconds()

            # Set the time of processing for this step, this should be UTC time is ISO 8601 format
            validation.time_processed = end_time.isoformat(timespec="milliseconds")

        self.time_end = datetime.datetime.now(datetime.timezone.utc)

        return self

    def all_passed(self) -> bool:
        """
        Determine if every validation step passed perfectly, with no failing test units.

        Returns
        -------
        bool
            `True` if all validations passed, `False` otherwise.
        """
        return all(validation.all_passed for validation in self.validation_info)

    def n(self, i: int | list[int] | None = None, scalar: bool = False) -> dict[int, int] | int:
        """
        Provides a dictionary of the number of test units for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the number of test units is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, int] | int
            A dictionary of the number of test units for each validation step or a scalar value.
        """
        result = self._get_validation_dict(i, "n")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def n_passed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, int] | int:
        """
        Provides a dictionary of the number of test units that passed for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the number of passing test units is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, int] | int
            A dictionary of the number of passing test units for each validation step or a scalar
            value.
        """
        result = self._get_validation_dict(i, "n_passed")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def n_failed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, int] | int:
        """
        Provides a dictionary of the number of test units that failed for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the number of failing test units is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, int] | int
            A dictionary of the number of failing test units for each validation step or a scalar
            value.
        """
        result = self._get_validation_dict(i, "n_failed")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def f_passed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, float] | float:
        """
        Provides a dictionary of the fraction of test units that passed for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the fraction of passing test units is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, float] | float
            A dictionary of the fraction of passing test units for each validation step or a scalar
            value.
        """
        result = self._get_validation_dict(i, "f_passed")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def f_failed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, float] | float:
        """
        Provides a dictionary of the fraction of test units that failed for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the fraction of failing test units is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, float] | float
            A dictionary of the fraction of failing test units for each validation step or a scalar
            value.
        """
        result = self._get_validation_dict(i, "f_failed")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def warn(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool:
        """
        Provides a dictionary of the warning status for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the warning status is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, bool] | bool
            A dictionary of the warning status for each validation step or a scalar value.
        """
        result = self._get_validation_dict(i, "warn")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def stop(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool:
        """
        Provides a dictionary of the stopping status for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the stopping status is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, bool] | bool
            A dictionary of the stopping status for each validation step or a scalar value.
        """
        result = self._get_validation_dict(i, "stop")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def notify(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool:
        """
        Provides a dictionary of the notification status for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the notification status is obtained.
            If `None`, all steps are included.
        scalar
            If `True` and `i` is a scalar, return the value as a scalar instead of a dictionary.

        Returns
        -------
        dict[int, bool] | bool
            A dictionary of the notification status for each validation step or a scalar value.
        """
        result = self._get_validation_dict(i, "notify")
        if scalar and isinstance(i, int):
            return result[i]
        return result

    def get_data_extracts(
        self, i: int | list[int] | None = None, frame: bool = False
    ) -> dict[int, FrameT | None] | FrameT | None:
        """
        Get the rows that failed for each validation step.

        After the `interrogate()` method has been called, the `get_data_extracts()` method can be
        used to extract the rows that failed in each row-based validation step (e.g.,
        `col_vals_gt()`, etc.). The method returns a dictionary of tables containing the rows that
        failed in every row-based validation function. If `frame=True` and `i` is a scalar, the
        value is conveniently returned as a table (forgoing the dictionary structure).

        Parameters
        ----------
        i
            The validation step number(s) from which the failed rows are obtained. If `None`, all
            steps are included.
        frame
            If `True` and `i` is a scalar, return the value as a DataFrame instead of a dictionary.

        Returns
        -------
        dict[int, FrameT | None] | FrameT | None
            A dictionary of tables containing the rows that failed in every row-based validation
            step or a DataFrame.

        Validation Methods that are Row-Based
        -------------------------------------
        The following validation methods are row-based and will have rows extracted when there are
        failing test units.

        - `col_vals_gt()`
        - `col_vals_ge()`
        - `col_vals_lt()`
        - `col_vals_le()`
        - `col_vals_eq()`
        - `col_vals_ne()`
        - `col_vals_between()`
        - `col_vals_outside()`
        - `col_vals_in_set()`
        - `col_vals_not_in_set()`
        - `col_vals_null()`
        - `col_vals_not_null()`
        - `col_vals_regex()`

        An extracted row means that a test unit failed for that row in the validation step. The
        extracted rows are a subset of the original table and are useful for further analysis or for
        understanding the nature of the failing test units.

        Examples
        --------
        Let's perform a series of validation steps on a Polars DataFrame. We'll use the
        `col_vals_gt()` in the first step, `col_vals_lt()` in the second step, and `col_vals_ge()`
        in the third step. The `interrogate()` method executes the validation; then, we can extract
        the rows that failed for each validation step.

        ```{python}
        import polars as pl
        import pointblank as pb

        tbl = pl.DataFrame(
            {
                "a": [5, 6, 5, 3, 6, 1],
                "b": [1, 2, 1, 5, 2, 6],
                "c": [3, 7, 2, 6, 3, 1],
            }
        )

        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=4)
            .col_vals_lt(columns="c", value=5)
            .col_vals_ge(columns="b", value=1)
            .interrogate()
        )

        validation.get_data_extracts()
        ```

        The `get_data_extracts()` method returns a dictionary of tables, where each table contains
        a subset of rows from the table. These are the rows that failed for each validation step.

        In the first step, the `col_vals_gt()` method was used to check if the values in column `a`
        were greater than `4`. The extracted table shows the rows where this condition was not met;
        look at the `a` column: all values are less than `4`.

        In the second step, the `col_vals_lt()` method was used to check if the values in column `c`
        were less than `5`. In the extracted two-row table, we see that the values in column `c` are
        greater than `5`.

        The third step (`col_vals_ge()`) checked if the values in column `b` were greater than or
        equal to `1`. There were no failing test units, so the extracted table is empty (i.e., has
        columns but no rows).

        The `i=` argument can be used to narrow down the extraction to one or more steps. For
        example, to extract the rows that failed in the first step only:

        ```{python}
        validation.get_data_extracts(i=1)
        ```

        Note that the first validation step is indexed at `1` (not `0`). This 1-based indexing is
        in place here to match the step numbers reported in the validation table. What we get back
        is still a dictionary, but it only contains one table (the one for the first step).

        If you want to get the extracted table as a DataFrame, set `frame=True` and provide a scalar
        value for `i`. For example, to get the extracted table for the second step as a DataFrame:

        ```{python}
        validation.get_data_extracts(i=2, frame=True)
        ```

        The extracted table is now a DataFrame, which can serve as a more convenient format for
        further analysis or visualization.
        """
        result = self._get_validation_dict(i, "extract")
        if frame and isinstance(i, int):
            return result[i]
        return result

    def get_json_report(
        self, use_fields: list[str] | None = None, exclude_fields: list[str] | None = None
    ) -> str:
        """
        Get a report of the validation results as a JSON-formatted string.

        Parameters
        ----------
        use_fields
            A list of fields to include in the report. If `None`, all fields are included.
        exclude_fields
            A list of fields to exclude from the report. If `None`, no fields are excluded.

        Returns
        -------
        str
            A JSON-formatted string representing the validation report.
        """

        if use_fields is not None and exclude_fields is not None:
            raise ValueError("Cannot specify both `use_fields=` and `exclude_fields=`.")

        if use_fields is None:
            fields = VALIDATION_REPORT_FIELDS
        else:

            # Ensure that the fields to use are valid
            _check_invalid_fields(use_fields, VALIDATION_REPORT_FIELDS)

            fields = use_fields

        if exclude_fields is not None:

            # Ensure that the fields to exclude are valid
            _check_invalid_fields(exclude_fields, VALIDATION_REPORT_FIELDS)

            fields = [field for field in fields if field not in exclude_fields]

        report = []

        for validation_info in self.validation_info:
            report_entry = {
                field: getattr(validation_info, field) for field in VALIDATION_REPORT_FIELDS
            }

            # If pre-processing functions are included in the report, convert them to strings
            if "pre" in fields:
                report_entry["pre"] = _pre_processing_funcs_to_str(report_entry["pre"])

            # Filter the report entry based on the fields to include
            report_entry = {field: report_entry[field] for field in fields}

            report.append(report_entry)

        return json.dumps(report, indent=4, default=str)

    def get_sundered_data(self, type="pass") -> FrameT:
        """
        Get the data that passed or failed the validation steps.

        Validation of the data is one thing but, sometimes, you want to use the best part of the
        input dataset for something else. The `get_sundered_data()` method works with a Validate
        object that has been interrogated (i.e., the `interrogate()` method was used). We can get
        either the 'pass' data piece (rows with no failing test units across all row-based
        validation functions), or, the 'fail' data piece (rows with at least one failing test unit
        across the same series of validations).

        Details
        -------
        There are some caveats to sundering. The validation steps considered for this splitting will
        only involve steps where:

        - of certain check types, where test units are cells checked row-by-row (e.g., the
        `col_vals_*()` methods)
        - `active=` is not set to `False`
        - `pre=` has not been given an expression for modify the input table

        So long as these conditions are met, the data will be split into two constituent tables: one
        with the rows that passed all validation steps and another with the rows that failed at
        least one validation step.

        Parameters
        ----------
        type
            The type of data to return. Options are `"pass"` or `"fail"`, where the former returns
            a table only containing rows where test units always passed validation steps, and the
            latter returns a table only containing rows had test units that failed in at least one
            validation step.

        Returns
        -------
        FrameT
            A table containing the data that passed or failed the validation steps.

        Examples
        --------
        Create a `Validate` plan of two validation steps, focused on testing row values for
        part of the `small_table` object. Then, use `interrogate()` to put the validation plan into
        action.

        """

        # Keep only the validation steps that:
        # - are row-based (included in `ROW_BASED_VALIDATION_TYPES`)
        # - are `active`
        validation_info = [
            validation
            for validation in self.validation_info
            if validation.assertion_type in ROW_BASED_VALIDATION_TYPES and validation.active
        ]

        # TODO: ensure that the stored evaluation tables across all steps have not been mutated
        # from the original table (via any `pre=` functions)

        # Obtain the validation steps that are to be used for sundering
        validation_steps_i = [validation.assertion_type for validation in validation_info]

        if len(validation_steps_i) == 0:

            if type == "pass":
                return self.data
            if type == "fail":
                return self.data[0:0]

        # Get an indexed version of the data
        # TODO: add argument for user to specify the index column name
        index_name = "pb_index_"

        data_nw = nw.from_native(self.data).with_row_index(name=index_name)

        # Get all validation step result tables and join together the `pb_is_good_` columns
        # ensuring that the columns are named uniquely (e.g., `pb_is_good_1`, `pb_is_good_2`, ...)
        # and that the index is reset
        for i, validation in enumerate(validation_info):

            results_tbl = nw.from_native(validation.tbl_checked)

            # Add row numbers to the results table
            results_tbl = results_tbl.with_row_index(name=index_name)

            # Add numerical suffix to the `pb_is_good_` column to make it unique
            results_tbl = results_tbl.select([index_name, "pb_is_good_"]).rename(
                {"pb_is_good_": f"pb_is_good_{i}"}
            )

            # Add the results table to the list of tables
            if i == 0:
                labeled_tbl_nw = results_tbl
            else:
                labeled_tbl_nw = labeled_tbl_nw.join(results_tbl, on=index_name, how="left")

        # Get list of columns that are the `pb_is_good_` columns
        pb_is_good_cols = [f"pb_is_good_{i}" for i in range(len(validation_steps_i))]

        # Determine the rows that passed all validation steps by checking if all `pb_is_good_`
        # columns are `True`
        labeled_tbl_nw = (
            labeled_tbl_nw.with_columns(pb_is_good_all=nw.all_horizontal(pb_is_good_cols))
            .join(data_nw, on=index_name, how="left")
            .drop(index_name)
        )

        bool_val = True if type == "pass" else False

        sundered_tbl = (
            labeled_tbl_nw.filter(nw.col("pb_is_good_all") == bool_val)
            .drop(pb_is_good_cols + ["pb_is_good_all"])
            .to_native()
        )

        return sundered_tbl

    def get_tabular_report(self, title: str | None = ":default:") -> GT:
        """
        Validation report as a GT table.

        Parameters
        ----------
        title
            Options for customizing the title of the report. The default is the `":default:"` value
            which produces a generic title. Another option is `":tbl_name:"`, and that presents the
            name of the table as the title for the report. If no title is wanted, then `":none:"`
            can be used. Aside from keyword options, text can be provided for the title. This will
            be interpreted as Markdown text and transformed internally to HTML.

        Returns
        -------
        GT
            A GT table object that represents the validation report.

        Examples
        --------
        Let's create a `Validate` object with a few validation steps and then interrogate the data
        table to see how it performs against the validation plan. We can then generate a tabular
        report to get a summary of the results.

        ```{python}
        import pointblank as pb
        import polars as pl

        # Create a Polars DataFrame
        tbl_pl = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7]})

        # Validate data using Polars DataFrame
        v = (
            pb.Validate(data=tbl_pl, tbl_name="tbl_xy", thresholds=(2, 3, 4))
            .col_vals_gt(columns="x", value=1)
            .col_vals_lt(columns="x", value=3)
            .col_vals_le(columns="y", value=7)
            .interrogate()
        )

        # Generate the tabular report
        v.get_tabular_report()
        ```

        The title option was set to `":default:"`, which produces a generic title for the report.
        We can change this to the name of the table by setting the title to `":tbl_name:"`. This
        will use the string provided in the `tbl_name=` argument of the `Validate` object.
        ```{python}
        v.get_tabular_report(title=":tbl_name:")
        ```
        """

        df_lib = _select_df_lib(preference="polars")

        # Get information on the input data table
        tbl_info = _get_tbl_type(data=self.data)

        # Get the thresholds object
        thresholds = self.thresholds

        # Convert the `validation_info` object to a dictionary
        validation_info_dict = _validation_info_as_dict(validation_info=self.validation_info)

        # Has the validation been performed? We can check the first `time_processed` entry in the
        # dictionary to see if it is `None` or not; The output of many cells in the reporting table
        # will be made blank if the validation has not been performed
        interrogation_performed = validation_info_dict.get("proc_duration_s", [None])[0] is not None

        # ------------------------------------------------
        # Process the `type_upd` entry
        # ------------------------------------------------

        # Add the `type_upd` entry to the dictionary
        validation_info_dict["type_upd"] = _transform_assertion_str(
            assertion_str=validation_info_dict["assertion_type"]
        )

        # ------------------------------------------------
        # Process the `values_upd` entry
        # ------------------------------------------------

        # Here, `values` will be transformed in ways particular to the assertion type (e.g.,
        # single values, ranges, sets, etc.)

        # Create a list to store the transformed values
        values_upd = []

        # Iterate over the values in the `values` entry
        values = validation_info_dict["values"]
        assertion_type = validation_info_dict["assertion_type"]
        inclusive = validation_info_dict["inclusive"]
        active = validation_info_dict["active"]

        for i, value in enumerate(values):

            # If the assertion type is a comparison of one value then add the value as a string
            if assertion_type[i] in [
                "col_vals_gt",
                "col_vals_lt",
                "col_vals_eq",
                "col_vals_ne",
                "col_vals_ge",
                "col_vals_le",
            ]:
                values_upd.append(str(value))

            # If the assertion type is a comparison of values within or outside of a range, add
            # the appropriate brackets (inclusive or exclusive) to the values
            elif assertion_type[i] in ["col_vals_between", "col_vals_outside"]:
                left_bracket = "[" if inclusive[i][0] else "("
                right_bracket = "]" if inclusive[i][1] else ")"
                values_upd.append(f"{left_bracket}{value[0]}, {value[1]}{right_bracket}")

            # If the assertion type is a comparison of a set of values; strip the leading and
            # trailing square brackets and single quotes
            elif assertion_type[i] in ["col_vals_in_set", "col_vals_not_in_set"]:
                values_upd.append(str(value)[1:-1].replace("'", ""))

            # If the assertion type checks for NULL or not NULL values, use an em dash
            elif assertion_type[i] in ["col_vals_null", "col_vals_not_null", "col_exists"]:
                values_upd.append("&mdash;")

            # If the assertion type is not recognized, add the value as a string
            else:
                values_upd.append(str(value))

        # Remove the `inclusive` entry from the dictionary
        validation_info_dict.pop("inclusive")

        # Add the `values_upd` entry to the dictionary
        validation_info_dict["values_upd"] = values_upd

        ## ------------------------------------------------
        ## The folloiwng entries rely on an interrogation
        ## to have been performed
        ## ------------------------------------------------

        # ------------------------------------------------
        # Add the `tbl` entry
        # ------------------------------------------------

        # Depending on if there was some pre-processing done, get the appropriate icon
        # for the table processing status to be displayed in the report under the `tbl` column

        validation_info_dict["tbl"] = _transform_tbl_preprocessed(
            pre=validation_info_dict["pre"], interrogation_performed=interrogation_performed
        )

        # ------------------------------------------------
        # Add the `eval` entry
        # ------------------------------------------------

        # Add the `eval` entry to the dictionary

        validation_info_dict["eval"] = _transform_eval(
            n=validation_info_dict["n"],
            interrogation_performed=interrogation_performed,
            active=active,
        )

        # ------------------------------------------------
        # Process the `test_units` entry
        # ------------------------------------------------

        # Add the `test_units` entry to the dictionary
        validation_info_dict["test_units"] = _transform_test_units(
            test_units=validation_info_dict["n"],
            interrogation_performed=interrogation_performed,
            active=active,
        )

        # ------------------------------------------------
        # Process `pass` and `fail` entries
        # ------------------------------------------------

        # Create a `pass` entry that concatenates the `n_passed` and `n_failed` entries (the length
        # of the `pass` entry should be equal to the length of the `n_passed` and `n_failed` entries)

        validation_info_dict["pass"] = _transform_passed_failed(
            n_passed_failed=validation_info_dict["n_passed"],
            f_passed_failed=validation_info_dict["f_passed"],
            interrogation_performed=interrogation_performed,
            active=active,
        )

        validation_info_dict["fail"] = _transform_passed_failed(
            n_passed_failed=validation_info_dict["n_failed"],
            f_passed_failed=validation_info_dict["f_failed"],
            interrogation_performed=interrogation_performed,
            active=active,
        )

        # ------------------------------------------------
        # Process `w_upd`, `s_upd`, `n_upd` entries
        # ------------------------------------------------

        # Transform `warn`, `stop`, and `notify` to `w_upd`, `s_upd`, and `n_upd` entries
        validation_info_dict["w_upd"] = _transform_w_s_n(
            values=validation_info_dict["warn"],
            color="#E5AB00",
            interrogation_performed=interrogation_performed,
        )
        validation_info_dict["s_upd"] = _transform_w_s_n(
            values=validation_info_dict["stop"],
            color="#CF142B",
            interrogation_performed=interrogation_performed,
        )
        validation_info_dict["n_upd"] = _transform_w_s_n(
            values=validation_info_dict["notify"],
            color="#439CFE",
            interrogation_performed=interrogation_performed,
        )

        # ------------------------------------------------
        # Process `status_color` entry
        # ------------------------------------------------

        # For the `status_color` entry, we will add a string based on the status of the validation:
        #
        # CASE 1: if `all_passed` is `True`, then the status color will be green
        # CASE 2: If none of `warn`, `stop`, or `notify` are `True`, then the status color will be
        #   light green ("#4CA64C66") (includes alpha of `0.5`)
        # CASE 3: If `warn` is `True`, then the status color will be yellow ("#FFBF00")
        # CASE 4: If `stop` is `True`, then the status color will be red (#CF142B")

        # Create a list to store the status colors
        status_color_list = []

        # Iterate over the validation steps
        for i in range(len(validation_info_dict["type_upd"])):
            if validation_info_dict["all_passed"][i]:
                status_color_list.append("#4CA64C")
            elif validation_info_dict["stop"][i]:
                status_color_list.append("#CF142B")
            elif validation_info_dict["warn"][i]:
                status_color_list.append("#FFBF00")
            else:
                # No status entered (W, S, N) but also not all passed
                status_color_list.append("#4CA64C66")

        # Add the `status_color` entry to the dictionary
        validation_info_dict["status_color"] = status_color_list

        # ------------------------------------------------
        # Process the extract entry
        # ------------------------------------------------

        # Create a list to store the extract colors
        extract_upd = []

        # Iterate over the validation steps
        for i in range(len(validation_info_dict["type_upd"])):

            # If the extract for this step is `None`, then produce an em dash then go to the next
            # iteration
            if validation_info_dict["extract"][i] is None:
                extract_upd.append("&mdash;")
                continue

            # If the extract for this step is not `None`, then produce a button that allows the
            # user to download the extract as a CSV file

            # Get the step number
            step_num = i + 1

            # Get the extract for this step
            extract = validation_info_dict["extract"][i]

            # Transform to Narwhals DataFrame
            extract_nw = nw.from_native(extract)

            # Get the number of rows in the extract
            n_rows = len(extract_nw)

            # If the number of rows is zero, then produce an em dash then go to the next iteration
            if n_rows == 0:
                extract_upd.append("&mdash;")
                continue

            # Write the CSV text
            csv_text = extract_nw.write_csv()

            # Use Base64 encoding to encode the CSV text
            csv_text_encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

            output_file_name = f"extract_{format(step_num, '04d')}.csv"

            # Create the download button
            button = (
                f'<a href="data:text/csv;base64,{csv_text_encoded}" download="{output_file_name}">'
                "<button "
                # TODO: Add a tooltip for the button
                #'aria-label="Download Extract" data-balloon-pos="left" '
                'style="background-color: #67C2DC; color: #FFFFFF; border: none; padding: 5px; '
                'font-weight: bold; cursor: pointer; border-radius: 4px;">CSV</button>'
                "</a>"
            )

            extract_upd.append(button)

        # Add the `extract_upd` entry to the dictionary
        validation_info_dict["extract_upd"] = extract_upd

        # Remove the `extract` entry from the dictionary
        validation_info_dict.pop("extract")

        # ------------------------------------------------
        # Removals from the dictionary
        # ------------------------------------------------

        # Remove the `assertion_type` entry from the dictionary
        validation_info_dict.pop("assertion_type")

        # Remove the `values` entry from the dictionary
        validation_info_dict.pop("values")

        # Remove the `n` entry from the dictionary
        validation_info_dict.pop("n")

        # Remove the `pre` entry from the dictionary
        validation_info_dict.pop("pre")

        # Remove the `proc_duration_s` entry from the dictionary
        validation_info_dict.pop("proc_duration_s")

        # Remove `n_passed`, `n_failed`, `f_passed`, and `f_failed` entries from the dictionary
        validation_info_dict.pop("n_passed")
        validation_info_dict.pop("n_failed")
        validation_info_dict.pop("f_passed")
        validation_info_dict.pop("f_failed")

        # Remove the `warn`, `stop`, and `notify` entries from the dictionary
        validation_info_dict.pop("warn")
        validation_info_dict.pop("stop")
        validation_info_dict.pop("notify")

        # Drop other keys from the dictionary
        validation_info_dict.pop("na_pass")
        validation_info_dict.pop("label")
        validation_info_dict.pop("brief")
        validation_info_dict.pop("active")
        validation_info_dict.pop("all_passed")

        # Create a table time string
        table_time = _create_table_time_html(time_start=self.time_start, time_end=self.time_end)

        # Create the title text
        title_text = _get_title_text(
            title=title, tbl_name=self.tbl_name, interrogation_performed=interrogation_performed
        )

        # Create the label, table type, and thresholds HTML fragments
        label_html = _create_label_html(label=self.label, start_time=self.time_start)
        table_type_html = _create_table_type_html(tbl_type=tbl_info, tbl_name=self.tbl_name)
        thresholds_html = _create_thresholds_html(thresholds=thresholds)

        # Compose the subtitle HTML fragment
        combined_subtitle = (
            "<div>"
            f"{label_html}"
            '<div style="padding-top: 10px; padding-bottom: 5px;">'
            f"{table_type_html}"
            f"{thresholds_html}"
            "</div>"
            "</div>"
        )

        # Create a DataFrame from the validation information using whatever the `df_lib` library is;
        # (it is either Polars or Pandas)
        df = df_lib.DataFrame(validation_info_dict)

        # Return the DataFrame as a Great Tables table
        gt_tbl = (
            GT(df, id="pb_tbl")
            .tab_header(title=html(title_text), subtitle=html(combined_subtitle))
            .tab_source_note(source_note=html(table_time))
            .fmt_markdown(columns=["pass", "fail", "extract_upd"])
            .opt_table_font(font=google_font(name="IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.css("height: 40px;"), locations=loc.body())
            .tab_style(
                style=style.text(weight="bold", color="#666666", size="13px"),
                locations=loc.body(columns="i"),
            )
            .tab_style(
                style=style.text(weight="bold", color="#666666"), locations=loc.column_labels()
            )
            .tab_style(
                style=style.text(size="28px", weight="bold", align="left", color="#444444"),
                locations=loc.title(),
            )
            .tab_style(
                style=style.text(
                    color="black", font=google_font(name="IBM Plex Mono"), size="11px"
                ),
                locations=loc.body(
                    columns=["type_upd", "column", "values_upd", "test_units", "pass", "fail"]
                ),
            )
            .tab_style(
                style=style.borders(sides="left", color="#E5E5E5", style="dashed"),
                locations=loc.body(columns=["column", "values_upd"]),
            )
            .tab_style(
                style=style.borders(
                    sides="left",
                    color="#E5E5E5",
                    style="dashed" if interrogation_performed else "none",
                ),
                locations=loc.body(columns=["pass", "fail"]),
            )
            .tab_style(
                style=style.fill(color="#FCFCFC" if interrogation_performed else "white"),
                locations=loc.body(columns=["w_upd", "s_upd", "n_upd"]),
            )
            .tab_style(
                style=style.borders(
                    sides="right",
                    color="#D3D3D3",
                    style="solid" if interrogation_performed else "none",
                ),
                locations=loc.body(columns="n_upd"),
            )
            .tab_style(
                style=style.borders(
                    sides="left",
                    color="#D3D3D3",
                    style="solid" if interrogation_performed else "none",
                ),
                locations=loc.body(columns="w_upd"),
            )
            .tab_style(
                style=style.fill(color="#FCFCFC" if interrogation_performed else "white"),
                locations=loc.body(columns=["tbl", "eval"]),
            )
            .tab_style(
                style=style.borders(
                    sides="right",
                    color="#D3D3D3",
                    style="solid" if interrogation_performed else "none",
                ),
                locations=loc.body(columns="eval"),
            )
            .tab_style(
                style=style.borders(sides="left", color="#D3D3D3", style="solid"),
                locations=loc.body(columns="tbl"),
            )
            .tab_style(
                style=style.fill(
                    color=from_column(column="status_color") if interrogation_performed else "white"
                ),
                locations=loc.body(columns="status_color"),
            )
            .tab_style(
                style=style.text(color="transparent", size="0px"),
                locations=loc.body(columns="status_color"),
            )
            .tab_style(
                style=style.css("white-space: nowrap; text-overflow: ellipsis; overflow: hidden;"),
                locations=loc.body(columns=["column", "values_upd"]),
            )
            .cols_label(
                cases={
                    "status_color": "",
                    "i": "",
                    "type_upd": "STEP",
                    "column": "COLUMNS",
                    "values_upd": "VALUES",
                    "tbl": "TBL",
                    "eval": "EVAL",
                    "test_units": "UNITS",
                    "pass": "PASS",
                    "fail": "FAIL",
                    "w_upd": "W",
                    "s_upd": "S",
                    "n_upd": "N",
                    "extract_upd": "EXT",
                }
            )
            .cols_width(
                cases={
                    "status_color": "4px",
                    "i": "35px",
                    "type_upd": "190px",
                    "column": "120px",
                    "values_upd": "120px",
                    "tbl": "50px",
                    "eval": "50px",
                    "test_units": "60px",
                    "pass": "60px",
                    "fail": "60px",
                    "w_upd": "30px",
                    "s_upd": "30px",
                    "n_upd": "30px",
                    "extract_upd": "65px",
                }
            )
            .cols_align(
                align="center", columns=["tbl", "eval", "w_upd", "s_upd", "n_upd", "extract_upd"]
            )
            .cols_align(align="right", columns=["test_units", "pass", "fail"])
            .cols_move_to_start(
                [
                    "status_color",
                    "i",
                    "type_upd",
                    "column",
                    "values_upd",
                    "tbl",
                    "eval",
                    "test_units",
                    "pass",
                    "fail",
                    "w_upd",
                    "s_upd",
                    "n_upd",
                    "extract_upd",
                ]
            )
            .tab_options(table_font_size="90%")
        )

        # If the interrogation has not been performed, then style the table columns dealing with
        # interrogation data as grayed out
        if not interrogation_performed:
            gt_tbl = gt_tbl.tab_style(
                style=style.fill(color="#F2F2F2"),
                locations=loc.body(
                    columns=["tbl", "eval", "test_units", "pass", "fail", "w_upd", "s_upd", "n_upd"]
                ),
            )

        # Transform `active` to a list of indices of inactive validations
        inactive_steps = [i for i, active in enumerate(active) if not active]

        # If there are inactive steps, then style those rows to be grayed out
        if inactive_steps:
            gt_tbl = gt_tbl.tab_style(
                style=style.fill(color="#F2F2F2"),
                locations=loc.body(rows=inactive_steps),
            )

        return gt_tbl

    def _add_validation(self, validation_info):
        """
        Add a validation to the list of validations.

        Parameters
        ----------
        validation_info
            Information about the validation to add.
        """

        validation_info.i = len(self.validation_info) + 1

        self.validation_info.append(validation_info)

        return self

    def _get_validation_dict(self, i: int | list[int] | None, attr: str) -> dict[int, int]:
        """
        Utility function to get a dictionary of validation attributes for each validation step.

        Parameters
        ----------
        i
            The validation step number(s) from which the attribute values are obtained.
            If `None`, all steps are included.
        attr
            The attribute name to retrieve from each validation step.

        Returns
        -------
        dict[int, int]
            A dictionary of the attribute values for each validation step.
        """
        if isinstance(i, int):
            i = [i]

        if i is None:
            return {validation.i: getattr(validation, attr) for validation in self.validation_info}

        return {
            validation.i: getattr(validation, attr)
            for validation in self.validation_info
            if validation.i in i
        }


def _validation_info_as_dict(validation_info: _ValidationInfo) -> dict:
    """
    Convert a `_ValidationInfo` object to a dictionary.

    Parameters
    ----------
    validation_info
        The `_ValidationInfo` object to convert to a dictionary.

    Returns
    -------
    dict
        A dictionary representing the `_ValidationInfo` object.
    """

    # Define the fields to include in the validation information
    validation_info_fields = [
        "i",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "pre",
        "label",
        "brief",
        "active",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warn",
        "stop",
        "notify",
        "extract",
        "proc_duration_s",
    ]

    # Filter the validation information to include only the selected fields
    validation_info_filtered = [
        {field: getattr(validation, field) for field in validation_info_fields}
        for validation in validation_info
    ]

    # Transform the validation information into a dictionary of lists so that it
    # can be used to create a DataFrame
    validation_info_dict = {field: [] for field in validation_info_fields}

    for validation in validation_info_filtered:
        for field in validation_info_fields:
            validation_info_dict[field].append(validation[field])

    return validation_info_dict


def _get_assertion_icon(icon: list[str], length_val: int = 30) -> list[str]:

    # For each icon, get the assertion icon SVG test from SVG_ICONS_FOR_ASSERTION_TYPES dictionary
    icon_svg = [SVG_ICONS_FOR_ASSERTION_TYPES.get(icon) for icon in icon]

    # Replace the width and height in the SVG string
    for i in range(len(icon_svg)):
        icon_svg[i] = _replace_svg_dimensions(icon_svg[i], height_width=length_val)

    return icon_svg


def _replace_svg_dimensions(svg: list[str], height_width: int | float) -> list[str]:

    svg = re.sub(r'width="[0-9]*?px', f'width="{height_width}px', svg)
    svg = re.sub(r'height="[0-9]*?px', f'height="{height_width}px', svg)

    return svg


def _get_title_text(title: str | None, tbl_name: str | None, interrogation_performed: bool) -> str:

    title = _process_title_text(title=title, tbl_name=tbl_name)

    if interrogation_performed:
        return title

    html_str = (
        "<div>"
        '<span style="float: left;">'
        f"{title}"
        "</span>"
        '<span style="float: right; text-decoration-line: underline; '
        "text-underline-position: under;"
        "font-size: 16px; text-decoration-color: #9C2E83;"
        'padding-top: 0.1em; padding-right: 0.4em;">'
        "No Interrogation Peformed"
        "</span>"
        "</div>"
    )

    return html_str


def _process_title_text(title: str | None, tbl_name: str | None) -> str:

    if title is None:
        title_text = ""
    elif title == ":default:":
        title_text = _get_default_title_text()
    elif title == ":none:":
        title_text = ""
    elif title == ":tbl_name:":
        if tbl_name is not None:
            title_text = f"<code>{tbl_name}</code>"
        else:
            title_text = ""
    else:
        title_text = commonmark.commonmark(title)

    return title_text


def _get_default_title_text() -> str:
    return "Pointblank Validation"


def _transform_tbl_preprocessed(pre: str, interrogation_performed: bool) -> list[str]:

    # If no interrogation was performed, return a list of empty strings
    if not interrogation_performed:
        return ["" for _ in range(len(pre))]

    # Iterate over the pre-processed table status and return the appropriate SVG icon name
    # (either 'unchanged' (None) or 'modified' (not None))
    status_list = []

    for status in pre:
        if status is None:
            status_list.append("unchanged")
        else:
            status_list.append("modified")

    return _get_preprocessed_table_icon(icon=status_list)


def _get_preprocessed_table_icon(icon: list[str]) -> list[str]:

    # For each icon, get the SVG icon from the SVG_ICONS_FOR_TBL_STATUS dictionary
    icon_svg = [SVG_ICONS_FOR_TBL_STATUS.get(icon) for icon in icon]

    return icon_svg


def _transform_eval(n: list[int], interrogation_performed: bool, active: list[bool]) -> list[str]:

    # If no interrogation was performed, return a list of empty strings
    if not interrogation_performed:
        return ["" for _ in range(len(n))]

    return [
        '<span style="color:#4CA64C;">&check;</span>' if active[i] else "&mdash;"
        for i in range(len(n))
    ]


def _transform_test_units(
    test_units: list[int], interrogation_performed: bool, active: list[bool]
) -> list[str]:

    # If no interrogation was performed, return a list of empty strings
    if not interrogation_performed:
        return ["" for _ in range(len(test_units))]

    return [
        (
            (
                str(test_units[i])
                if test_units[i] < 10000
                else str(vals.fmt_number(test_units[i], n_sigfig=3, compact=True)[0])
            )
            if active[i]
            else "&mdash;"
        )
        for i in range(len(test_units))
    ]


def _fmt_lg(value: int) -> str:
    return vals.fmt_number(value, n_sigfig=3, compact=True)[0]


def _transform_passed_failed(
    n_passed_failed: list[int],
    f_passed_failed: list[float],
    interrogation_performed: bool,
    active: list[bool],
) -> list[str]:

    if not interrogation_performed:
        return ["" for _ in range(len(n_passed_failed))]

    passed_failed = [
        (
            f"{n_passed_failed[i] if n_passed_failed[i] < 10000 else _fmt_lg(n_passed_failed[i])}"
            f"<br />{vals.fmt_number(f_passed_failed[i], decimals=2)[0]}"
            if active[i]
            else "&mdash;"
        )
        for i in range(len(n_passed_failed))
    ]

    return passed_failed


def _transform_w_s_n(values, color, interrogation_performed):

    # If no interrogation was performed, return a list of empty strings
    if not interrogation_performed:
        return ["" for _ in range(len(values))]

    return [
        (
            "&mdash;"
            if value is None
            else (
                f'<span style="color: {color};">&#9679;</span>'
                if value is True
                else f'<span style="color: {color};">&cir;</span>' if value is False else value
            )
        )
        for value in values
    ]


def _transform_assertion_str(assertion_str: list[str]) -> list[str]:

    # Get the SVG icons for the assertion types
    svg_icon = _get_assertion_icon(icon=assertion_str)

    # Append `()` to the `assertion_str`
    assertion_str = [x + "()" for x in assertion_str]

    # Obtain the number of characters contained in the assertion
    # string; this is important for sizing components appropriately
    assertion_type_nchar = [len(x) for x in assertion_str]

    # Declare the text size based on the length of `assertion_str`
    text_size = [10 if nchar + 2 >= 20 else 11 for nchar in assertion_type_nchar]

    # Create the assertion type update using a list comprehension
    type_upd = [
        f"""
        <div style="margin:0;padding:0;display:inline-block;height:30px;vertical-align:middle;">
        <!--?xml version="1.0" encoding="UTF-8"?-->{svg}
        </div>
        <span style="font-family: 'IBM Plex Mono', monospace, courier; color: black; font-size:{size}px;"> {assertion}</span>
        """
        for assertion, svg, size in zip(assertion_str, svg_icon, text_size)
    ]

    return type_upd


def _pre_processing_funcs_to_str(pre: Callable) -> str | list[str]:

    if isinstance(pre, Callable):
        return _get_callable_source(fn=pre)


def _get_callable_source(fn: Callable) -> str:
    if isinstance(fn, Callable):
        try:
            source_lines, _ = inspect.getsourcelines(fn)
            source = "".join(source_lines).strip()
            # Extract the `pre` argument from the source code
            pre_arg = _extract_pre_argument(source)
            return pre_arg
        except (OSError, TypeError):  # pragma: no cover
            return fn.__name__
    return fn


def _extract_pre_argument(source: str) -> str:

    # Find the start of the `pre` argument
    pre_start = source.find("pre=")
    if pre_start == -1:
        return source

    # Find the end of the `pre` argument
    pre_end = source.find(",", pre_start)
    if pre_end == -1:
        pre_end = len(source)

    # Extract the `pre` argument and remove the leading `pre=`
    pre_arg = source[pre_start + len("pre=") : pre_end].strip()

    return pre_arg


def _create_table_time_html(
    time_start: datetime.datetime | None, time_end: datetime.datetime | None
) -> str:

    if time_start is None:
        return ""

    # Get the time duration (difference between `time_end` and `time_start`) in seconds
    time_duration = (time_end - time_start).total_seconds()

    # If the time duration is less than 1 second, use a simplified string, otherwise
    # format the time duration to four decimal places
    if time_duration < 1:
        time_duration_fmt = "< 1 s"
    else:
        time_duration_fmt = f"{time_duration:.4f} s"

    # Format the start time and end time in the format: "%Y-%m-%d %H:%M:%S %Z"
    time_start_fmt = time_start.strftime("%Y-%m-%d %H:%M:%S %Z")
    time_end_fmt = time_end.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Generate an HTML string that displays the start time, duration, and end time
    return (
        f"<div style='margin-top: 5px; margin-bottom: 5px;'>"
        f"<span style='background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: "
        f"inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: "
        f"solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: "
        f"2px 10px 2px 10px;'>{time_start_fmt}</span>"
        f"<span style='background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: "
        f"inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: "
        f"tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;'>{time_duration_fmt}</span>"
        f"<span style='background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: "
        f"inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; "
        f"font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;'>"
        f"{time_end_fmt}</span>"
        f"</div>"
    )


def _create_label_html(label: str | None, start_time: str) -> str:

    if label is None:

        # Remove the decimal and everything beyond that
        start_time = str(start_time).split(".")[0]

        # Replace the space character with a pipe character
        start_time = start_time.replace(" ", "|")

        label = start_time

    return (
        f"<span style='text-decoration-style: solid; text-decoration-color: #ADD8E6; "
        f"text-decoration-line: underline; text-underline-position: under; color: #333333; "
        f"font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; "
        f"padding-right: 2px;'>{label}</span>"
    )


def _create_table_type_html(tbl_type: str | None, tbl_name: str | None) -> str:

    if tbl_type is None:
        return ""

    style = TABLE_TYPE_STYLES.get(tbl_type)

    if style is None:
        return ""

    if tbl_name is None:
        return (
            f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
            f"position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px {style['background']}; "
            f"font-weight: bold; padding: 2px 10px 2px 10px; font-size: smaller;'>{style['label']}</span>"
        )

    return (
        f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;'>{style['label']}</span>"
        f"<span style='background-color: none; color: #222222; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 10px 5px -4px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;'>{tbl_name}</span>"
    )


def _create_thresholds_html(thresholds: Thresholds) -> str:

    if thresholds == Thresholds():
        return ""

    warn = (
        thresholds.warn_fraction
        if thresholds.warn_fraction is not None
        else (thresholds.warn_count if thresholds.warn_count is not None else "&mdash;")
    )

    stop = (
        thresholds.stop_fraction
        if thresholds.stop_fraction is not None
        else (thresholds.stop_count if thresholds.stop_count is not None else "&mdash;")
    )

    notify = (
        thresholds.notify_fraction
        if thresholds.notify_fraction is not None
        else (thresholds.notify_count if thresholds.notify_count is not None else "&mdash;")
    )

    return (
        "<span>"
        '<span style="background-color: #E5AB00; color: white; padding: 0.5em 0.5em; position: inherit; '
        "text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #E5AB00; font-weight: bold; "
        'padding: 2px 15px 2px 15px; font-size: smaller;">WARN</span>'
        '<span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; '
        "margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #E5AB00; padding: 2px 15px 2px 15px; "
        'font-size: smaller; margin-right: 5px;">'
        f"{warn}"
        "</span>"
        '<span style="background-color: #D0182F; color: white; padding: 0.5em 0.5em; position: inherit; '
        "text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #D0182F; font-weight: bold; "
        'padding: 2px 15px 2px 15px; font-size: smaller;">STOP</span>'
        '<span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; '
        "margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #D0182F; padding: 2px 15px 2px 15px; "
        'font-size: smaller; margin-right: 5px;">'
        f"{stop}"
        "</span>"
        '<span style="background-color: #499FFE; color: white; padding: 0.5em 0.5em; position: inherit; '
        "text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #499FFE; font-weight: bold; "
        'padding: 2px 15px 2px 15px; font-size: smaller;">NOTIFY</span>'
        '<span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; '
        "margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #499FFE; padding: 2px 15px 2px 15px; "
        'font-size: smaller;">'
        f"{notify}"
        "</span>"
        "</span>"
    )


def _get_tbl_type(data: FrameT) -> str:

    type_str = str(type(data))

    ibis_tbl = "ibis.expr.types.relations.Table" in type_str

    # TODO: in a later release of Narwhals, there will be a method for getting the namespace:
    # `get_native_namespace()`

    if not ibis_tbl:

        df_ns_str = str(nw.from_native(data).__native_namespace__())

        # Detect through regex if the table is a polars or pandas DataFrame
        if re.search(r"polars", df_ns_str, re.IGNORECASE):
            return "polars"
        elif re.search(r"pandas", df_ns_str, re.IGNORECASE):
            return "pandas"

    # If ibis is present, then get the table's backend name
    ibis_present = _is_lib_present(lib_name="ibis")

    if ibis_present:
        import ibis

        # TODO: Getting the backend 'name' is currently a bit brittle right now; as it is,
        #       we either extract the backend name from the table name or get the backend name
        #       from the get_backend() method and name attribute

        backend = ibis.get_backend(data).name

        # Try using the get_name() method to get the table name, this is important for elucidating
        # the original table type since it sometimes gets handled by duckdb

        if backend == "duckdb":

            try:
                tbl_name = data.get_name()
            except AttributeError:
                tbl_name = None

            if tbl_name is not None:

                if "memtable" in tbl_name:
                    return "memtable"

                if "read_parquet" in tbl_name:
                    return "parquet"

            else:
                return "duckdb"

        return backend

    return "unknown"


def _is_lib_present(lib_name: str) -> bool:
    import importlib

    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False


def _select_df_lib(preference: str = "polars") -> Any:

    # Determine whether Pandas or Polars is available
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import polars as pl
    except ImportError:
        pl = None

    # If neither Pandas nor Polars is available, raise an ImportError
    if pd is None and pl is None:
        raise ImportError(
            "Generating a report with the `get_tabular_report()` method requires either the "
            "Polars or the Pandas library to be installed."
        )

    # Return the library based on preference, if both are available
    if pd is not None and pl is not None:
        if preference == "polars":
            return pl
        else:
            return pd

    return pl if pl is not None else pd
