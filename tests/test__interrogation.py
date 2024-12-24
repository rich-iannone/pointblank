import pytest
import pandas as pd
import polars as pl

from pointblank._interrogation import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
    ColValsRegex,
    ColExistsHasType,
    RowsDistinct,
)


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": ["4", "5", "6", "7"], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": ["4", "5", "6", "7"], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pd_distinct():
    return pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d"],
            "col_2": ["a", "a", "c", "d"],
            "col_3": ["a", "a", "d", "e"],
        }
    )


@pytest.fixture
def tbl_pl_distinct():
    return pl.DataFrame(
        {
            "col_1": ["a", "b", "c", "d"],
            "col_2": ["a", "a", "c", "d"],
            "col_3": ["a", "a", "d", "e"],
        }
    )


COLUMN_LIST = ["x", "y", "z", "pb_is_good_"]

COLUMN_LIST_DISTINCT = ["col_1", "col_2", "col_3", "pb_is_good_"]


@pytest.mark.parametrize(
    "tbl_fixture, assertion_method",
    [
        ("tbl_pd", "gt"),
        ("tbl_pd", "lt"),
        ("tbl_pd", "eq"),
        ("tbl_pd", "ne"),
        ("tbl_pd", "ge"),
        ("tbl_pd", "le"),
        ("tbl_pl", "gt"),
        ("tbl_pl", "lt"),
        ("tbl_pl", "eq"),
        ("tbl_pl", "ne"),
        ("tbl_pl", "ge"),
        ("tbl_pl", "le"),
    ],
)
def test_col_vals_compare_one(request, tbl_fixture, assertion_method):

    tbl = request.getfixturevalue(tbl_fixture)

    col_vals_compare_one = ColValsCompareOne(
        data_tbl=tbl,
        column="x",
        value=1,
        na_pass=True,
        threshold=10,
        assertion_method=assertion_method,
        allowed_types=["numeric"],
    )

    assert isinstance(
        col_vals_compare_one.test_unit_res,
        pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame,
    )

    if tbl_fixture == "tbl_pd":
        assert col_vals_compare_one.test_unit_res.columns.tolist() == COLUMN_LIST
        assert col_vals_compare_one.get_test_results().columns.tolist() == COLUMN_LIST
    else:
        assert col_vals_compare_one.test_unit_res.columns == COLUMN_LIST
        assert col_vals_compare_one.get_test_results().columns == COLUMN_LIST

    assert col_vals_compare_one.test() is True


def test_col_vals_compare_one_invalid_assertion_method(tbl_pd):

    with pytest.raises(ValueError):
        ColValsCompareOne(
            data_tbl=tbl_pd,
            column="x",
            value=1,
            na_pass=True,
            threshold=10,
            assertion_method="invalid",
            allowed_types=["numeric"],
        )


@pytest.mark.parametrize(
    "tbl_fixture, assertion_method",
    [
        ("tbl_pd", "between"),
        ("tbl_pd", "outside"),
        ("tbl_pl", "between"),
        ("tbl_pl", "outside"),
    ],
)
def test_col_vals_compare_two(request, tbl_fixture, assertion_method):

    tbl = request.getfixturevalue(tbl_fixture)

    col_vals_compare_two = ColValsCompareTwo(
        data_tbl=tbl,
        column="x",
        value1=1,
        value2=2,
        inclusive=(True, True),
        na_pass=True,
        threshold=10,
        assertion_method=assertion_method,
        allowed_types=["numeric"],
    )

    assert isinstance(
        col_vals_compare_two.test_unit_res,
        pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame,
    )

    if tbl_fixture == "tbl_pd":
        assert col_vals_compare_two.test_unit_res.columns.tolist() == COLUMN_LIST
        assert col_vals_compare_two.get_test_results().columns.tolist() == COLUMN_LIST
    else:
        assert col_vals_compare_two.test_unit_res.columns == COLUMN_LIST
        assert col_vals_compare_two.get_test_results().columns == COLUMN_LIST

    assert col_vals_compare_two.test() is True


def test_col_vals_compare_two_invalid_assertion_method(tbl_pd):

    with pytest.raises(ValueError):
        ColValsCompareTwo(
            data_tbl=tbl_pd,
            column="x",
            value1=1,
            value2=2,
            inclusive=(True, True),
            na_pass=True,
            threshold=10,
            assertion_method="invalid",
            allowed_types=["numeric"],
        )


@pytest.mark.parametrize(
    "tbl_fixture, inside",
    [
        ("tbl_pd", True),
        ("tbl_pd", False),
        ("tbl_pl", True),
        ("tbl_pl", False),
    ],
)
def test_col_vals_compare_set(request, tbl_fixture, inside):

    tbl = request.getfixturevalue(tbl_fixture)

    col_vals_compare_set = ColValsCompareSet(
        data_tbl=tbl,
        column="x",
        values=[1, 2, 3],
        threshold=10,
        inside=inside,
        allowed_types=["numeric"],
    )

    assert isinstance(
        col_vals_compare_set.test_unit_res,
        pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame,
    )

    if tbl_fixture == "tbl_pd":
        assert col_vals_compare_set.test_unit_res.columns.tolist() == COLUMN_LIST
        assert col_vals_compare_set.get_test_results().columns.tolist() == COLUMN_LIST
    else:
        assert col_vals_compare_set.test_unit_res.columns == COLUMN_LIST
        assert col_vals_compare_set.get_test_results().columns == COLUMN_LIST

    assert col_vals_compare_set.test() is True


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_regex(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    col_vals_regex = ColValsRegex(
        data_tbl=tbl,
        column="y",
        pattern=r"^[0-9]$",
        na_pass=True,
        threshold=10,
        allowed_types=["str"],
    )

    assert isinstance(
        col_vals_regex.test_unit_res,
        pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame,
    )

    if tbl_fixture == "tbl_pd":
        assert col_vals_regex.test_unit_res.columns.tolist() == COLUMN_LIST
        assert col_vals_regex.get_test_results().columns.tolist() == COLUMN_LIST
    else:
        assert col_vals_regex.test_unit_res.columns == COLUMN_LIST
        assert col_vals_regex.get_test_results().columns == COLUMN_LIST

    assert col_vals_regex.test() is True


@pytest.mark.parametrize(
    "tbl_fixture, assertion_method",
    [
        ("tbl_pd", "exists"),
        ("tbl_pl", "exists"),
    ],
)
def test_column_exists_has_type(request, tbl_fixture, assertion_method):

    tbl = request.getfixturevalue(tbl_fixture)

    col_exists_has_type = ColExistsHasType(
        data_tbl=tbl,
        column="x",
        threshold=1,
        assertion_method=assertion_method,
    )

    assert isinstance(col_exists_has_type.test_unit_res, int)
    assert col_exists_has_type.test_unit_res == 1


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd_distinct", "tbl_pl_distinct"])
def test_rows_distinct(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    rows_distinct = RowsDistinct(
        data_tbl=tbl,
        columns_subset=["col_1", "col_2", "col_3"],
        threshold=1,
    )

    assert isinstance(
        rows_distinct.test_unit_res,
        pd.DataFrame if tbl_fixture == "tbl_pd_distinct" else pl.DataFrame,
    )

    if tbl_fixture == "tbl_pd_distinct":
        assert rows_distinct.test_unit_res.columns.tolist() == COLUMN_LIST_DISTINCT
        assert rows_distinct.get_test_results().columns.tolist() == COLUMN_LIST_DISTINCT
    else:
        assert rows_distinct.test_unit_res.columns == COLUMN_LIST_DISTINCT
        assert rows_distinct.get_test_results().columns == COLUMN_LIST_DISTINCT
