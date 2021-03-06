test_that("the `has_columns()` function works when used directly with data", {
  
  # Expect TRUE when *all* of the given column names is present
  expect_true(small_table %>% has_columns("date"))
  expect_true(small_table %>% has_columns(c("a", "b")))
  expect_true(small_table %>% has_columns(vars(a, b)))
  expect_true(small_table %>% has_columns(c(vars(a, b), vars(c))))
  
  # Expect FALSE when *any* of the given column names is absent 
  expect_false(small_table %>% has_columns(vars(a, h)))
  expect_false(small_table %>% has_columns(vars(h, j)))
  
  # Select helper functions don't work with `has_columns()` and will
  # always throw errors
  expect_error(small_table %>% has_columns(pointblank::contains("te")))
  expect_error(small_table %>% has_columns(pointblank::starts_with("date")))
  expect_error(small_table %>% has_columns(pointblank::ends_with("time")))
  expect_error(small_table %>% has_columns(pointblank::everything()))
  expect_error(small_table %>% has_columns(pointblank::matches(".*te")))
  
  # Expect that using inputs that are not tabular result in errors
  expect_error(has_columns(list(a = "2"), "a"))
  expect_error(has_columns(c(a = 1, b = 2), "a"))
  expect_error(has_columns(matrix(c(1, 2, 3, 4), nrow = 2), "b"))
})
