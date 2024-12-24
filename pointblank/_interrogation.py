from __future__ import annotations
from dataclasses import dataclass

from typing import Any

import narwhals as nw
from narwhals.typing import FrameT
from narwhals.dependencies import is_pandas_dataframe, is_polars_dataframe

from pointblank._utils import _column_test_prep, _column_subset_test_prep, _convert_to_narwhals
from pointblank.thresholds import _threshold_check
from pointblank._constants import IBIS_BACKENDS
from pointblank.column import Column


@dataclass
class Interrogator:
    """
    Compare values against a single value, a set of values, or a range of values.

    Parameters
    ----------
    x
        The values to compare.
    column
        The column to check.
    columns_subset
        The subset of columns to use for the check.
    compare
        The value to compare against. Used in the following interrogations:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to
    set
        The set of values to compare against. Used in the following interrogations:
        - 'isin' for values in the set
        - 'notin' for values not in the set
    pattern
        The regular expression pattern to compare against. Used in the following:
        - 'regex' for values that match the pattern
    low
        The lower bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    high
        The upper bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (low, high). Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    tbl_type
        The type of table to use for the assertion. This is used to determine the backend for the
        assertion. The default is 'local' but it can also be any of the table types in the
        `IBIS_BACKENDS` constant.

    Returns
    -------
    list[bool]
        A list of booleans where `True` indicates a passing test unit.
    """

    x: nw.DataFrame | Any
    column: str = None
    columns_subset: list[str] = None
    compare: float | int | list[float | int] = None
    set: list[float | int] = None
    pattern: str = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None
    inclusive: tuple[bool, bool] = None
    na_pass: bool = False
    tbl_type: str = "local"

    def gt(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] > self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            else:

                tbl = self.x.mutate(
                    pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] > ibis.literal(self.compare),
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

        # Local backends (Narwhals) ---------------------------------

        compare_expr = _get_compare_expr_nw(compare=self.compare)

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
                pb_is_good_3=nw.col(self.column) > compare_expr,
            )
            .with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )
            .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
            .to_native()
        )

    def lt(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] < self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            else:

                tbl = self.x.mutate(
                    pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] < ibis.literal(self.compare),
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

        # Local backends (Narwhals) ---------------------------------

        compare_expr = _get_compare_expr_nw(compare=self.compare)

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
                pb_is_good_3=nw.col(self.column) < compare_expr,
            )
            .with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )
            .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
            .to_native()
        )

    def eq(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] == self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            else:

                tbl = self.x.mutate(
                    pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] == ibis.literal(self.compare),
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

        # Local backends (Narwhals) ---------------------------------

        if isinstance(self.compare, Column):

            compare_expr = _get_compare_expr_nw(compare=self.compare)

            tbl = self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
            )

            tbl = tbl.with_columns(
                pb_is_good_3=(~nw.col(self.compare.name).is_null() & ~nw.col(self.column).is_null())
            )

            if is_pandas_dataframe(tbl.to_native()):

                tbl = tbl.with_columns(
                    pb_is_good_4=nw.col(self.column) - compare_expr,
                )

                tbl = tbl.with_columns(
                    pb_is_good_=nw.col("pb_is_good_1")
                    | nw.col("pb_is_good_2")
                    | (nw.col("pb_is_good_4") == 0 & ~nw.col("pb_is_good_3").is_null())
                )

            else:

                tbl = tbl.with_columns(
                    pb_is_good_4=nw.col(self.column) == compare_expr,
                )

                tbl = tbl.with_columns(
                    pb_is_good_=nw.col("pb_is_good_1")
                    | nw.col("pb_is_good_2")
                    | (nw.col("pb_is_good_4") & ~nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2"))
                )

            return tbl.drop(
                "pb_is_good_1", "pb_is_good_2", "pb_is_good_3", "pb_is_good_4"
            ).to_native()

        else:
            compare_expr = _get_compare_expr_nw(compare=self.compare)

            tbl = self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
            )

            tbl = tbl.with_columns(pb_is_good_3=nw.col(self.column) == compare_expr)

            tbl = tbl.with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )

            return tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

    def ne(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] != self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=ibis.ifelse(
                    self.x[self.column].notnull(),
                    self.x[self.column] != ibis.literal(self.compare),
                    ibis.literal(False),
                ),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        # Local backends (Narwhals) ---------------------------------

        # Determine if the reference and comparison columns have any null values
        ref_col_has_null_vals = _column_has_null_values(table=self.x, column=self.column)

        if isinstance(self.compare, Column):
            compare_name = self.compare.name if isinstance(self.compare, Column) else self.compare
            cmp_col_has_null_vals = _column_has_null_values(table=self.x, column=compare_name)
        else:
            cmp_col_has_null_vals = False

        # If neither column has null values, we can proceed with the comparison
        # without too many complications
        if not ref_col_has_null_vals and not cmp_col_has_null_vals:

            if isinstance(self.compare, Column):

                compare_expr = _get_compare_expr_nw(compare=self.compare)

                return self.x.with_columns(
                    pb_is_good_=nw.col(self.column) != compare_expr,
                ).to_native()

            else:

                return self.x.with_columns(
                    pb_is_good_=nw.col(self.column) != nw.lit(self.compare),
                ).to_native()

        # If either column has null values, we need to handle the comparison
        # much more carefully since we can't inadverdently compare null values
        # to non-null values

        if isinstance(self.compare, Column):

            compare_expr = _get_compare_expr_nw(compare=self.compare)

            # CASE 1: the reference column has null values but the comparison column does not
            if ref_col_has_null_vals and not cmp_col_has_null_vals:

                if is_pandas_dataframe(self.x.to_native()):

                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.lit(self.column) != nw.col(self.compare.name),
                    )

                else:

                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.col(self.column) != nw.col(self.compare.name),
                    )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                    )

                if is_polars_dataframe(self.x.to_native()):

                    # There may be Null values in the pb_is_good_2 column, change those to
                    # True if na_pass is True, False otherwise

                    tbl = tbl.with_columns(
                        pb_is_good_2=nw.when(nw.col("pb_is_good_2").is_null())
                        .then(False)
                        .otherwise(nw.col("pb_is_good_2")),
                    )

                    if self.na_pass:

                        tbl = tbl.with_columns(
                            pb_is_good_2=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                    .drop("pb_is_good_1", "pb_is_good_2")
                    .to_native()
                )

            # CASE 2: the comparison column has null values but the reference column does not
            elif not ref_col_has_null_vals and cmp_col_has_null_vals:

                if is_pandas_dataframe(self.x.to_native()):

                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column) != nw.lit(self.compare.name),
                        pb_is_good_2=nw.col(self.compare.name).is_null(),
                    )

                else:

                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column) != nw.col(self.compare.name),
                        pb_is_good_2=nw.col(self.compare.name).is_null(),
                    )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_1=nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2")
                    )

                if is_polars_dataframe(self.x.to_native()):

                    if self.na_pass:

                        tbl = tbl.with_columns(
                            pb_is_good_1=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_1"))
                    .drop("pb_is_good_1", "pb_is_good_2")
                    .to_native()
                )

            # CASE 3: both columns have null values and there may potentially be cases where
            # there could even be null/null comparisons
            elif ref_col_has_null_vals and cmp_col_has_null_vals:

                tbl = self.x.with_columns(
                    pb_is_good_1=nw.col(self.column).is_null(),
                    pb_is_good_2=nw.col(self.compare.name).is_null(),
                    pb_is_good_3=nw.col(self.column) != nw.col(self.compare.name),
                )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_3=nw.col("pb_is_good_3")
                        & ~nw.col("pb_is_good_1")
                        & ~nw.col("pb_is_good_2")
                    )

                if is_polars_dataframe(self.x.to_native()):

                    if self.na_pass:

                        tbl = tbl.with_columns(
                            pb_is_good_3=(
                                nw.when(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                                .then(True)
                                .otherwise(False)
                            )
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_3"))
                    .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
                    .to_native()
                )

        else:

            # Case where the reference column contains null values
            if ref_col_has_null_vals:

                # Create individual cases for Pandas and Polars

                if is_pandas_dataframe(self.x.to_native()):
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.lit(self.column) != nw.lit(self.compare),
                    )

                    if not self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                        )

                    return (
                        tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                        .drop("pb_is_good_1", "pb_is_good_2")
                        .to_native()
                    )

                elif is_polars_dataframe(self.x.to_native()):

                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
                        pb_is_good_2=nw.lit(self.na_pass),  # Pass if any Null in val or compare
                    )

                    tbl = tbl.with_columns(pb_is_good_3=nw.col(self.column) != nw.lit(self.compare))

                    tbl = tbl.with_columns(
                        pb_is_good_=(
                            (
                                (nw.col("pb_is_good_1") & nw.col("pb_is_good_2"))
                                | (nw.col("pb_is_good_3") & ~nw.col("pb_is_good_1"))
                            )
                        )
                    )

                    tbl = tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

                    return tbl

    def ge(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] >= self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] >= ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        # Local backends (Narwhals) ---------------------------------

        compare_expr = _get_compare_expr_nw(compare=self.compare)

        tbl = self.x.with_columns(
            pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
            pb_is_good_2=(
                nw.col(self.compare.name).is_null() & self.na_pass
                if isinstance(self.compare, Column)
                else nw.lit(False)
            ),
            pb_is_good_3=nw.col(self.column) >= compare_expr,
        ).with_columns(
            pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
        )

        return tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

    def le(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.compare, Column):

                tbl = self.x.mutate(
                    pb_is_good_1=(self.x[self.column].isnull() | self.x[self.compare.name].isnull())
                    & ibis.literal(self.na_pass),
                    pb_is_good_2=self.x[self.column] <= self.x[self.compare.name],
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                    "pb_is_good_1", "pb_is_good_2"
                )

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] <= ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        # Local backends (Narwhals) ---------------------------------

        compare_expr = _get_compare_expr_nw(compare=self.compare)

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
                pb_is_good_3=nw.col(self.column) <= compare_expr,
            )
            .with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )
            .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
            .to_native()
        )

    def between(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.low, Column) or isinstance(self.high, Column):

                if isinstance(self.low, Column):
                    low_val = self.x[self.low.name]
                else:
                    low_val = ibis.literal(self.low)

                if isinstance(self.high, Column):
                    high_val = self.x[self.high.name]
                else:
                    high_val = ibis.literal(self.high)

                if isinstance(self.low, Column) and isinstance(self.high, Column):
                    tbl = self.x.mutate(
                        pb_is_good_1=(
                            self.x[self.column].isnull()
                            | self.x[self.low.name].isnull()
                            | self.x[self.high.name].isnull()
                        )
                        & ibis.literal(self.na_pass)
                    )
                elif isinstance(self.low, Column):
                    tbl = self.x.mutate(
                        pb_is_good_1=(self.x[self.column].isnull() | self.x[self.low.name].isnull())
                        & ibis.literal(self.na_pass)
                    )
                elif isinstance(self.high, Column):
                    tbl = self.x.mutate(
                        pb_is_good_1=(
                            self.x[self.column].isnull() | self.x[self.high.name].isnull()
                        )
                        & ibis.literal(self.na_pass)
                    )

                if self.inclusive[0]:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] >= low_val)
                else:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] > low_val)

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(tbl.pb_is_good_2.notnull(), tbl.pb_is_good_2, False)
                )

                if self.inclusive[1]:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] <= high_val)
                else:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] < high_val)

                tbl = tbl.mutate(
                    pb_is_good_3=ibis.ifelse(tbl.pb_is_good_3.notnull(), tbl.pb_is_good_3, False)
                )

                return tbl.mutate(
                    pb_is_good_=tbl.pb_is_good_1 | (tbl.pb_is_good_2 & tbl.pb_is_good_3)
                ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

            else:

                low_val = ibis.literal(self.low)
                high_val = ibis.literal(self.high)

                tbl = self.x.mutate(
                    pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass)
                )

                if self.inclusive[0]:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] >= low_val)
                else:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] > low_val)

                if self.inclusive[1]:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] <= high_val)
                else:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] < high_val)

                return tbl.mutate(
                    pb_is_good_=tbl.pb_is_good_1 | (tbl.pb_is_good_2 & tbl.pb_is_good_3)
                ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

        # Local backends (Narwhals) ---------------------------------

        low_val = _get_compare_expr_nw(compare=self.low)
        high_val = _get_compare_expr_nw(compare=self.high)

        tbl = self.x.with_columns(
            pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
            pb_is_good_2=(  # lb is Null in Column
                nw.col(self.low.name).is_null() if isinstance(self.low, Column) else nw.lit(False)
            ),
            pb_is_good_3=(  # ub is Null in Column
                nw.col(self.high.name).is_null() if isinstance(self.high, Column) else nw.lit(False)
            ),
            pb_is_good_4=nw.lit(self.na_pass),  # Pass if any Null in lb, val, or ub
        )

        if self.inclusive[0]:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) >= low_val)
        else:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) > low_val)

        if self.inclusive[1]:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) <= high_val)
        else:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) < high_val)

        tbl = (
            tbl.with_columns(
                pb_is_good_=(
                    (
                        (nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3"))
                        & nw.col("pb_is_good_4")
                    )
                    | (nw.col("pb_is_good_5") & nw.col("pb_is_good_6"))
                )
            )
            .drop(
                "pb_is_good_1",
                "pb_is_good_2",
                "pb_is_good_3",
                "pb_is_good_4",
                "pb_is_good_5",
                "pb_is_good_6",
            )
            .to_native()
        )

        return tbl

    def outside(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            if isinstance(self.low, Column) or isinstance(self.high, Column):

                if isinstance(self.low, Column):
                    low_val = self.x[self.low.name]
                else:
                    low_val = ibis.literal(self.low)

                if isinstance(self.high, Column):
                    high_val = self.x[self.high.name]
                else:
                    high_val = ibis.literal(self.high)

                if isinstance(self.low, Column) and isinstance(self.high, Column):

                    tbl = self.x.mutate(
                        pb_is_good_1=(
                            self.x[self.column].isnull()
                            | self.x[self.low.name].isnull()
                            | self.x[self.high.name].isnull()
                        )
                        & ibis.literal(self.na_pass)
                    )

                elif isinstance(self.low, Column):
                    tbl = self.x.mutate(
                        pb_is_good_1=(self.x[self.column].isnull() | self.x[self.low.name].isnull())
                        & ibis.literal(self.na_pass)
                    )
                elif isinstance(self.high, Column):
                    tbl = self.x.mutate(
                        pb_is_good_1=(
                            self.x[self.column].isnull() | self.x[self.high.name].isnull()
                        )
                        & ibis.literal(self.na_pass)
                    )

                if self.inclusive[0]:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] < low_val)
                else:
                    tbl = tbl.mutate(pb_is_good_2=tbl[self.column] <= low_val)

                if self.inclusive[1]:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] > high_val)
                else:
                    tbl = tbl.mutate(pb_is_good_3=tbl[self.column] >= high_val)

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(
                        tbl.pb_is_good_3.isnull(),
                        False,
                        tbl.pb_is_good_2,
                    )
                )

                tbl = tbl.mutate(
                    pb_is_good_3=ibis.ifelse(
                        tbl.pb_is_good_2.isnull(),
                        False,
                        tbl.pb_is_good_3,
                    )
                )

                tbl = tbl.mutate(
                    pb_is_good_2=ibis.ifelse(
                        tbl.pb_is_good_2.isnull(),
                        False,
                        tbl.pb_is_good_2,
                    )
                )

                tbl = tbl.mutate(
                    pb_is_good_3=ibis.ifelse(
                        tbl.pb_is_good_3.isnull(),
                        False,
                        tbl.pb_is_good_3,
                    )
                )

                return tbl.mutate(
                    pb_is_good_=tbl.pb_is_good_1 | (tbl.pb_is_good_2 | tbl.pb_is_good_3)
                ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

            low_val = ibis.literal(self.low)
            high_val = ibis.literal(self.high)

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass)
            )

            if self.inclusive[0]:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] < low_val)
            else:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] <= low_val)

            if self.inclusive[1]:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] > high_val)
            else:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] >= high_val)

            return tbl.mutate(
                pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2 | tbl.pb_is_good_3
            ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

        # Local backends (Narwhals) ---------------------------------

        low_val = _get_compare_expr_nw(compare=self.low)
        high_val = _get_compare_expr_nw(compare=self.high)

        tbl = self.x.with_columns(
            pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
            pb_is_good_2=(  # lb is Null in Column
                nw.col(self.low.name).is_null() if isinstance(self.low, Column) else nw.lit(False)
            ),
            pb_is_good_3=(  # ub is Null in Column
                nw.col(self.high.name).is_null() if isinstance(self.high, Column) else nw.lit(False)
            ),
            pb_is_good_4=nw.lit(self.na_pass),  # Pass if any Null in lb, val, or ub
        )

        if self.inclusive[0]:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) < low_val)
        else:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) <= low_val)

        if self.inclusive[1]:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) > high_val)
        else:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) >= high_val)

        tbl = tbl.with_columns(
            pb_is_good_5=nw.when(nw.col("pb_is_good_5").is_null())
            .then(False)
            .otherwise(nw.col("pb_is_good_5")),
            pb_is_good_6=nw.when(nw.col("pb_is_good_6").is_null())
            .then(False)
            .otherwise(nw.col("pb_is_good_6")),
        )

        tbl = (
            tbl.with_columns(
                pb_is_good_=(
                    (
                        (nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3"))
                        & nw.col("pb_is_good_4")
                    )
                    | (
                        (nw.col("pb_is_good_5") & ~nw.col("pb_is_good_3"))
                        | (nw.col("pb_is_good_6")) & ~nw.col("pb_is_good_2")
                    )
                )
            )
            .drop(
                "pb_is_good_1",
                "pb_is_good_2",
                "pb_is_good_3",
                "pb_is_good_4",
                "pb_is_good_5",
                "pb_is_good_6",
            )
            .to_native()
        )

        return tbl

    def isin(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(pb_is_good_=self.x[self.column].isin(self.set))

        # Local backends (Narwhals) ---------------------------------

        return self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_in(self.set),
        ).to_native()

    def notin(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(pb_is_good_=self.x[self.column].notin(self.set))

        # Local backends (Narwhals) ---------------------------------

        return (
            self.x.with_columns(
                pb_is_good_=nw.col(self.column).is_in(self.set),
            )
            .with_columns(pb_is_good_=~nw.col("pb_is_good_"))
            .to_native()
        )

    def regex(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column].re_search(self.pattern),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        # Local backends (Narwhals) ---------------------------------

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.when(~nw.col(self.column).is_null())
                .then(nw.col(self.column).str.contains(pattern=self.pattern))
                .otherwise(False),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def null(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(
                pb_is_good_=self.x[self.column].isnull(),
            )

        # Local backends (Narwhals) ---------------------------------

        return self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_null(),
        ).to_native()

    def not_null(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(
                pb_is_good_=~self.x[self.column].isnull(),
            )

        # Local backends (Narwhals) ---------------------------------

        return self.x.with_columns(
            pb_is_good_=~nw.col(self.column).is_null(),
        ).to_native()

    def rows_distinct(self) -> FrameT | Any:

        # Ibis backends ---------------------------------------------

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x

            # Get the column subset to use for the test
            if self.columns_subset is None:
                columns_subset = tbl.columns
            else:
                columns_subset = self.columns_subset

            # Create a subset of the table with only the columns of interest and count the
            # number of times each unique row (or portion thereof) appears
            tbl = tbl.group_by(columns_subset).mutate(pb_count_=ibis._.count())

            # Passing rows will have the value `1` (no duplicates, so True), otherwise False applies
            return tbl.mutate(pb_is_good_=tbl["pb_count_"] == 1).drop("pb_count_")

        # Local backends (Narwhals) ---------------------------------

        tbl = self.x

        # Get the column subset to use for the test
        if self.columns_subset is None:
            columns_subset = tbl.columns
        else:
            columns_subset = self.columns_subset

        # Create a subset of the table with only the columns of interest
        subset_tbl = tbl.select(columns_subset)

        # Check for duplicates in the subset table, creating a series of booleans
        pb_is_good_series = subset_tbl.is_duplicated()

        # Add the series to the input table
        tbl = tbl.with_columns(pb_is_good_=~pb_is_good_series)

        return tbl.to_native()


@dataclass
class ColValsCompareOne:
    """
    Compare values in a table column against a single value.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    value
        A value to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('gt' for greater than, 'lt' for less than).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value: float | int
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "gt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).gt()
        elif self.assertion_method == "lt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).lt()
        elif self.assertion_method == "eq":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).eq()
        elif self.assertion_method == "ne":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ne()
        elif self.assertion_method == "ge":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ge()
        elif self.assertion_method == "le":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).le()
        elif self.assertion_method == "null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).null()
        elif self.assertion_method == "not_null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).not_null()
        else:
            raise ValueError(
                """Invalid comparison type. Use:
                - `gt` for greater than,
                - `lt` for less than,
                - `eq` for equal to,
                - `ne` for not equal to,
                - `ge` for greater than or equal to,
                - `le` for less than or equal to,
                - `null` for null values, or
                - `not_null` for not null values.
                """
            )

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareTwo:
    """
    General routine to compare values in a column against two values.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    value1
        A value to check against.
    value2
        A value to check against.
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (value1, value2).
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('between' for between two values and 'outside' for outside two
        values).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value1: float | int
    value2: float | int
    inclusive: tuple[bool, bool]
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "between":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).between()
        elif self.assertion_method == "outside":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).outside()
        else:
            raise ValueError(
                """Invalid assertion type. Use:
                - `between` for values between two values, or
                - `outside` for values outside two values."""
            )

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareSet:
    """
    General routine to compare values in a column against a set of values.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    values
        A set of values to check against.
    threshold
        The maximum number of failing test units to allow.
    inside
        `True` to check if the values are inside the set, `False` to check if the values are
        outside the set.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    values: list[float | int]
    threshold: int
    inside: bool
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.inside:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).isin()
        else:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).notin()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsRegex:
    """
    Check if values in a column match a regular expression pattern.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    pattern
        The regular expression pattern to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    pattern: str
    na_pass: bool
    threshold: int
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            column=self.column,
            pattern=self.pattern,
            na_pass=self.na_pass,
            tbl_type=self.tbl_type,
        ).regex()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColExistsHasType:
    """
    Check if a column exists in a DataFrame or has a certain data type.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('exists' for column existence).
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    threshold: int
    assertion_method: str
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _convert_to_narwhals(df=self.data_tbl)

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        if self.assertion_method == "exists":

            res = int(self.column in tbl.columns)

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class RowsDistinct:
    """
    Check if rows in a DataFrame are distinct.

    Parameters
    ----------
    data_tbl
        A data table.
    columns_subset
        A list of columns to check for distinctness.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    columns_subset: list[str] | None
    threshold: int
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_subset_test_prep(df=self.data_tbl, columns_subset=self.columns_subset)

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            columns_subset=self.columns_subset,
            tbl_type=self.tbl_type,
        ).rows_distinct()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class NumberOfTestUnits:
    """
    Count the number of test units in a column.
    """

    df: FrameT
    column: str

    def get_test_units(self, tbl_type: str) -> int:

        if tbl_type == "pandas" or tbl_type == "polars":

            # Convert the DataFrame to a format that narwhals can work with and:
            #  - check if the column exists
            dfn = _column_test_prep(
                df=self.df, column=self.column, allowed_types=None, check_exists=False
            )

            return len(dfn)

        if tbl_type in IBIS_BACKENDS:

            # Get the count of test units and convert to a native format
            # TODO: check whether pandas or polars is available
            return self.df.count().to_polars()


def _get_compare_expr_nw(compare: Any) -> Any:
    if isinstance(compare, Column):
        return nw.col(compare.name)
    return compare


def _column_has_null_values(table: FrameT, column: str) -> bool:
    null_count = (table.select(column).null_count())[column][0]

    if null_count is None or null_count == 0:
        return False

    return True
