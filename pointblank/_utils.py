from __future__ import annotations

import re
import inspect

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._constants import ASSERTION_TYPE_METHOD_MAP, GENERAL_COLUMN_TYPES


def _convert_to_narwhals(df: FrameT) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with
    return nw.from_native(df)


def _check_column_exists(dfn: nw.DataFrame, column: str) -> None:
    """
    Check if a column exists in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column to check for existence.

    Raises
    ------
    ValueError
        When the column is not found in the DataFrame.
    """

    if column not in dfn.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")


def _is_numeric_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a numeric type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is numeric, `False` otherwise.
    """
    # Define the regular expression pattern for numeric data types
    numeric_pattern = re.compile(r"^(int|float)\d*$")
    return bool(numeric_pattern.match(dtype))


def _is_date_or_datetime_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a date or datetime type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is date or datetime, `False` otherwise.
    """
    # Define the regular expression pattern for date or datetime data types
    date_pattern = re.compile(r"^(date|datetime).*$")
    return bool(date_pattern.match(dtype))


def _is_duration_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a duration type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is a duration, `False` otherwise.
    """
    # Define the regular expression pattern for duration data types
    duration_pattern = re.compile(r"^duration.*$")
    return bool(duration_pattern.match(dtype))


def _get_column_dtype(
    dfn: nw.DataFrame, column: str, raw: bool = False, lowercased: bool = True
) -> str:
    """
    Get the data type of a column in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column from which to get the data type.
    raw
        If `True`, return the raw data type string.
    lowercased
        If `True`, return the data type string in lowercase.

    Returns
    -------
    str
        The data type of the column.
    """

    if raw:  # pragma: no cover
        return dfn.collect_schema().get(column)

    column_dtype_str = str(dfn.collect_schema().get(column))

    if lowercased:
        return column_dtype_str.lower()

    return column_dtype_str


def _check_column_type(dfn: nw.DataFrame, column: str, allowed_types: list[str]) -> None:
    """
    Check if a column is of a certain data type.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column to check for data type.
    dtype
        The data type to check for. These are shorthand types and the following are supported:
        - `"numeric"`: Numeric data types (`int`, `float`)
        - `"str"`: String data type
        - `"bool"`: Boolean data type
        - `"datetime"`: Date or Datetime data type
        - `"duration"`: Duration data type

    Raises
    ------
    TypeError
        When the column is not of the specified data type.
    """

    # Get the data type of the column as a lowercase string
    column_dtype = str(dfn.collect_schema().get(column)).lower()

    # If `allowed_types` is empty, raise a ValueError
    if not allowed_types:
        raise ValueError("No allowed types specified.")

    # If any of the supplied `allowed_types` are not in the `GENERAL_COLUMN_TYPES` list,
    # raise a ValueError
    _check_invalid_fields(fields=allowed_types, valid_fields=GENERAL_COLUMN_TYPES)

    if _is_numeric_dtype(dtype=column_dtype) and "numeric" not in allowed_types:
        raise TypeError(f"Column '{column}' is numeric.")

    if column_dtype == "string" and "str" not in allowed_types:
        raise TypeError(f"Column '{column}' is a string.")

    if column_dtype == "boolean" and "bool" not in allowed_types:
        raise TypeError(f"Column '{column}' is a boolean.")

    if _is_date_or_datetime_dtype(dtype=column_dtype) and "datetime" not in allowed_types:
        raise TypeError(f"Column '{column}' is a date or datetime.")

    if _is_duration_dtype(dtype=column_dtype) and "duration" not in allowed_types:
        raise TypeError(f"Column '{column}' is a duration.")


def _column_test_prep(
    df: FrameT, column: str, allowed_types: list[str] | None, check_exists: bool = True
) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check if the column exists
    if check_exists:
        _check_column_exists(dfn=dfn, column=column)

    # Check if the column is of the allowed types. Raise a TypeError if not.
    if allowed_types:
        _check_column_type(dfn=dfn, column=column, allowed_types=allowed_types)

    return dfn


def _column_subset_test_prep(
    df: FrameT, columns_subset: list[str] | None, check_exists: bool = True
) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check whether all columns exist
    if check_exists and columns_subset:

        for column in columns_subset:
            _check_column_exists(dfn=dfn, column=column)

    return dfn


def _get_fn_name() -> str:

    # Get the current function name
    fn_name = inspect.currentframe().f_back.f_code.co_name

    return fn_name


def _get_assertion_from_fname() -> str:

    # Get the current function name
    func_name = inspect.currentframe().f_back.f_code.co_name

    # Use the `ASSERTION_TYPE_METHOD_MAP` dictionary to get the assertion type
    assertion = ASSERTION_TYPE_METHOD_MAP.get(func_name)

    return assertion


def _check_invalid_fields(fields: list[str], valid_fields: list[str]):
    """
    Check if any fields in the list are not in the valid fields list.

    Parameters
    ----------
    fields
        The list of fields to check.
    valid_fields
        The list of valid fields.

    Raises
    ------
    ValueError
        If any field in the list is not in the valid fields list.
    """
    for field in fields:
        if field not in valid_fields:
            raise ValueError(f"Invalid field: {field}")
