#
#                _         _    _      _                _    
#               (_)       | |  | |    | |              | |   
#  _ __    ___   _  _ __  | |_ | |__  | |  __ _  _ __  | | __
# | '_ \  / _ \ | || '_ \ | __|| '_ \ | | / _` || '_ \ | |/ /
# | |_) || (_) || || | | || |_ | |_) || || (_| || | | ||   < 
# | .__/  \___/ |_||_| |_| \__||_.__/ |_| \__,_||_| |_||_|\_\
# | |                                                        
# |_|                                                        
# 
# This file is part of the 'rich-iannone/pointblank' package.
# 
# (c) Richard Iannone <riannone@me.com>
# 
# For full copyright and license information, please look at
# https://rich-iannone.github.io/pointblank/LICENSE.html
#


# Get the column names for a table
# - works across all supported data sources
# - returns 'character' of length n
get_table_column_names <- function(data) {
  
  colnames(data)
}

# Get the total number of rows in a table
# - works across all supported data sources
# - returns 'numeric' of length 1
get_table_total_rows <- function(data) {

  as.numeric(dplyr::pull(dplyr::count(data, name = "n"), n))
}

# Get the total number of rows in a table
# - works across all supported data sources
# - returns 'integer' of length 1
get_table_total_columns <- function(data) {

  ncol(data)
}

# Get the total number of NA/NULLs in a table
# - works across all supported data sources
# - returns 'numeric' of length 1
get_table_total_missing_values <- function(data) {
  
  collected <- 
    dplyr::collect(
      dplyr::summarise_all(
        data, ~ sum(ifelse(is.na(.), 1, 0), na.rm = TRUE)
      )
    )

  sum(as.vector(t(collected)))
}

# Get the total number of distinct rows in a table
# - works across all supported data sources
# - returns 'numeric' of length 1
get_table_total_distinct_rows <- function(data) {

  as.numeric(dplyr::pull(dplyr::count(dplyr::distinct(data), name = "n"), n))
}

# Get the total number of distinct rows in a table column
# (must be a table with a single column)
# - works across all supported data sources
# - returns 'integer' of length 1
get_table_column_distinct_rows <- function(data_column) {
  
  as.integer(get_table_total_distinct_rows(data = data_column))
}

# Get the total number of NA/NULLs in a table column
# (must be a table with a single column)
# - works across all supported data sources
# - returns 'numeric' of length 1
get_table_column_na_values <- function(data_column) {
  
  get_table_total_missing_values(data = data_column)
}

# Get the total number of Inf/-Inf values in a table column
# (must be a table with a single column)
# - works across all supported data sources
# - returns 'integer' of length 1
get_table_column_inf_values <- function(data_column) {
  
  if (!inherits(data_column, "tbl_dbi") &&
      !inherits(data_column, "tbl_spark")) {
    
    inf_cells <-
      data_column %>%
      dplyr::pull(1) %>%
      is.infinite() %>%
      sum()
    
  } else {
    inf_cells <- 0L
  }
  
  inf_cells
}

# Get a 1-row tibble of mean, min, and max values for a table column
# (must be a table with a single column)
# - works across all supported data sources
# - returns 'tibble' of 1 row, 3 columns
get_table_column_summary <- function(data_column, round = 2) {
  
  data_column %>%
    dplyr::summarize_all(
      .funs = list(
        ~ mean(., na.rm = TRUE),
        ~ min(., na.rm = TRUE),
        ~ max(., na.rm = TRUE)
      )
    ) %>%
    dplyr::collect() %>%
    dplyr::summarize_all(~ round(., round)) %>%
    dplyr::mutate_all(.funs = as.numeric)
}

# Get a list of quantile statistics for a table column
# (must be a table with a single column)
# - works only with local data frames
# - returns a 'list' with 9 elements
get_df_column_qtile_stats <- function(data_column) {
  
  data_column %>%
    dplyr::summarize_all(
      .funs = list(
        min = ~ min(., na.rm = TRUE),
        p05 = ~ unname(stats::quantile(., probs = 0.05, na.rm = TRUE)),
        q_1 = ~ unname(stats::quantile(., probs = 0.25, na.rm = TRUE)),
        med = ~ unname(stats::quantile(., probs = 0.50, na.rm = TRUE)),
        q_3 = ~ unname(stats::quantile(., probs = 0.75, na.rm = TRUE)),
        p95 = ~ unname(stats::quantile(., probs = 0.95, na.rm = TRUE)),
        max = ~ max(., na.rm = TRUE),
        iqr = ~ stats::IQR(., na.rm = TRUE)
      )
    ) %>%
    dplyr::mutate(range = max - min) %>%
    dplyr::summarize_all(~ round(., 2)) %>%
    as.list()
}

# Get the mean value of a column in a table
# - works only with `tbl_dbi` objects
# - returns 'numeric' of length 1
get_dbi_column_mean <- function(data_column) {
  
  x <- dplyr::rename(data_column, a = 1)
  x <- dplyr::group_by(x)
  x <- dplyr::summarize(x, "__mean__" = mean(a, na.rm = TRUE))
  x <- dplyr::pull(x, `__mean__`)
  
  x
}

# Get the mean value of a column in a table
# - works only with `tbl_spark` objects
# - returns 'numeric' of length 1
get_spark_column_mean <- function(data_column) {
  
  get_dbi_column_mean(data_column = data_column)
}

# Get the variance value of a column in a table, after obtaining the mean value
# - works only with `tbl_dbi` objects
# - returns 'numeric' of length 1
get_dbi_column_variance <- function(data_column, mean_value) {
  
  x <- dplyr::rename(data_column, a = 1)
  x <- dplyr::mutate(x, "__diff__" = (!!mean_value - a)^2)
  x <- dplyr::group_by(x)
  x <- dplyr::summarize(x, "__var__"  = mean(`__diff__`, na.rm = TRUE))
  x <- dplyr::pull(x, `__var__`)
  
  x
}

# Get the variance value of a column in a table, after obtaining the mean value
# - works only with `tbl_spark` objects
# - returns 'numeric' of length 1
get_spark_column_variance <- function(data_column, mean_value) {
  
  get_dbi_column_variance(data_column = data_column, mean_value = mean_value)
}

# Get a list of quantile statistics for a Spark DataFrame column
# (must be a table with a single column)
# - works only with Spark DataFrames
# - returns a 'list' with 9 elements
get_spark_column_qtile_stats <- function(data_column) {
  
  column_name <- colnames(data_column)
  
  quantiles <- 
    unname(
      sparklyr::sdf_quantile(
        data_column, column_name,
        probabilities = c(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)
      )
    )

  list(
    min = quantiles[1],
    p05 = quantiles[2],
    q_1 = quantiles[3],
    med = quantiles[4],
    q_3 = quantiles[5],
    p95 = quantiles[6],
    max = quantiles[7],
    iqr = quantiles[5] - quantiles[3],
    range = quantiles[7] - quantiles[1]
  ) %>% 
    lapply(FUN = function(x) round(x, 2))
}

# Get a list of quantile statistics for a DBI table column
# (must be a table with a single column)
# - works only with `tbl_dbi` objects
# - returns a 'list' with 9 elements
get_dbi_column_qtile_stats <- function(data_column) {
  
  data_arranged <- 
    data_column %>%
    dplyr::rename(a = 1) %>%
    dplyr::filter(!is.na(a)) %>%
    dplyr::arrange(a) %>%
    utils::head(6E8)
  
  n_rows <- get_table_total_rows(data = data_arranged)
  
  quantile_rows <- floor(c(0.05, 0.25, 0.5, 0.75, 0.95) * n_rows)
  
  # Should there be any rows estimated as `0`, change those to `1`
  quantile_rows[quantile_rows == 0] <- 1
  
  dplyr::tibble(
    min = data_arranged %>% 
      dplyr::summarize(a = min(a, na.rm = TRUE)) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    p05 = data_arranged %>% 
      utils::head(quantile_rows[1]) %>%
      dplyr::arrange(dplyr::desc(a)) %>%
      utils::head(1) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    q_1 = data_arranged %>% 
      utils::head(quantile_rows[2]) %>%
      dplyr::arrange(dplyr::desc(a)) %>%
      utils::head(1) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    med = data_arranged %>% 
      utils::head(quantile_rows[3]) %>%
      dplyr::arrange(dplyr::desc(a)) %>%
      utils::head(1) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    q_3 = data_arranged %>%
      utils::head(quantile_rows[4]) %>%
      dplyr::arrange(dplyr::desc(a)) %>%
      utils::head(1) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    p95 = data_arranged %>%
      utils::head(quantile_rows[5]) %>%
      dplyr::arrange(dplyr::desc(a)) %>%
      utils::head(1) %>%
      dplyr::pull(a) %>%
      as.numeric(),
    max = data_arranged %>%
      dplyr::summarize(a = max(a, na.rm = TRUE)) %>%
      dplyr::pull(a) %>%
      as.numeric()
  ) %>%
    dplyr::mutate(
      iqr = q_3 - q_1,
      range = max - min
    ) %>%
    dplyr::summarize_all(~ round(., 2)) %>%
    as.list()
}

get_table_column_nchar_stats <- function(data_column) {
  
  data_column %>%
    dplyr::mutate_all(.funs = nchar) %>%
    dplyr::rename(nchar = 1) %>%
    dplyr::summarize_all(
      .funs = list(
        mean = ~ mean(., na.rm = TRUE),
        min = ~ min(., na.rm = TRUE),
        max = ~ max(., na.rm = TRUE)
      )
    ) %>%
    dplyr::collect() %>%
    dplyr::mutate_all(.funs = as.numeric) %>%
    as.list()
}

# Get a ggplot2 histogram plot of character lengths for a table column
# (must be a table with a single column)
# - works across all supported data sources
# - returns a ggplot object
get_table_column_histogram <- function(data_column, lang, locale) {
  
  # TODO: Use locale value to get proper settings comma format
  
  x_label <- get_lsv("table_scan/plot_lab_string_length")[[lang]]
  y_label <- get_lsv("table_scan/plot_lab_count")[[lang]]
  
  suppressWarnings(
    data_column %>%
      dplyr::mutate_all(.funs = nchar) %>%
      dplyr::rename(nchar = 1) %>%
      dplyr::group_by(nchar) %>%
      dplyr::summarize(n = dplyr::n()) %>%
      dplyr::collect() %>%
      dplyr::filter(!is.na(nchar)) %>%
      dplyr::mutate_all(.funs = as.numeric) %>%
      ggplot2::ggplot(ggplot2::aes(x = nchar, y = n)) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::geom_hline(yintercept = 0, color = "#B2B2B2") +
      ggplot2::labs(x = x_label, y = y_label) +
      ggplot2::scale_x_continuous(limits = c(0, NA)) +
      ggplot2::scale_y_continuous(labels = scales::comma_format()) +
      ggplot2::theme_minimal()
  )
}

# Get a tibble of binned missing value proportions for a table
# - works only with `tbl_dbi` objects
# - returns 'tibble' of 20 bins * y columns rows, 4 columns
get_tbl_dbi_missing_tbl <- function(data) {
  
  n_rows <- get_table_total_rows(data = data)
  col_names <- get_table_column_names(data = data)
  
  if (n_rows < 20) {
    n_breaks <- n_rows
  } else {
    n_breaks <- 20
  }
  
  cuts <- floor(seq(from = 1, to = n_rows, length.out = n_breaks + 1))[-1]
  bin_size <- cuts[1]
  
  frequency_list <- 
    lapply(
      col_names,
      FUN = function(x__) {
        
        col_num <- which(col_names %in% x__)
        
        if (length(cuts) < 20) {
          bin_num <- cuts
        } else {
          bin_num <- 1:20
        }
        
        missing_tally <- 0L
        
        missing_freq <- 
          vapply(
            bin_num,
            FUN.VALUE = numeric(1),
            FUN = function(x) {
              
              missing_n_span <- 
                data %>% 
                dplyr::select(1, dplyr::one_of(x__))
              
              if (ncol(missing_n_span) == 1) {
                
                missing_n_span <- dplyr::rename(missing_n_span, a = 1)
                
              } else {
                
                missing_n_span <- dplyr::rename(missing_n_span, a = 2)
              }
              
              missing_n_span <- 
                missing_n_span %>%
                utils::head(cuts[x]) %>%
                dplyr::summarize_all(
                  ~ sum(ifelse(is.na(.), 1, 0), na.rm = TRUE)
                ) %>%
                dplyr::pull(a) %>%
                as.integer()
              
              missing_bin <- missing_n_span - missing_tally
              
              missing_tally <<- missing_tally + missing_bin
              
              as.numeric(missing_bin / bin_size)
            }
          )
        
        dplyr::tibble(
          col_name = x__,
          col_num = col_num,
          bin_num = bin_num,
          value = missing_freq
        )
      }
    )
  
  frequency_list %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(value = ifelse(value == 0, NA_real_, value)) %>%
    dplyr::mutate(col_name = factor(col_name, levels = colnames(data)))
}

# Get a tibble of binned missing value proportions for a table
# - works only with `tbl_spark` objects
# - returns 'tibble' of 20 bins * y columns rows, 4 columns
get_tbl_spark_missing_tbl <- function(data) {
  
  get_tbl_dbi_missing_tbl(data = data)
}

# Get a tibble of binned missing value proportions for a table
# - works only with data frame objects
# - returns 'tibble' of 20 bins * y columns rows, 4 columns
get_tbl_df_missing_tbl <- function(data) {
  
  n_rows <- get_table_total_rows(data = data)
  col_names <- get_table_column_names(data = data)
  
  if (n_rows < 20) {
    n_breaks <- n_rows
  } else {
    n_breaks <- 20
  }
  
  cuts_ <- floor(seq(from = 1, to = n_rows, length.out = n_breaks + 1))[-1]
  
  frequency_list <- 
    lapply(
      col_names,
      FUN = function(x__) {
        
        data <- dplyr::select(data, dplyr::one_of(x__))
        data <- tibble::rowid_to_column(data)
        data <- 
          dplyr::mutate(data, `::cut_group::` = dplyr::case_when(
            rowid < cuts_[1]  ~ 1L,
            rowid < cuts_[2]  ~ 2L,
            rowid < cuts_[3]  ~ 3L,
            rowid < cuts_[4]  ~ 4L,
            rowid < cuts_[5]  ~ 5L,
            rowid < cuts_[6]  ~ 6L,
            rowid < cuts_[7]  ~ 7L,
            rowid < cuts_[8]  ~ 8L,
            rowid < cuts_[9]  ~ 9L,
            rowid < cuts_[10] ~ 10L,
            rowid < cuts_[11] ~ 11L,
            rowid < cuts_[12] ~ 12L,
            rowid < cuts_[13] ~ 13L,
            rowid < cuts_[14] ~ 14L,
            rowid < cuts_[15] ~ 15L,
            rowid < cuts_[16] ~ 16L,
            rowid < cuts_[17] ~ 17L,
            rowid < cuts_[18] ~ 18L,
            rowid < cuts_[19] ~ 19L,
            rowid < cuts_[20] ~ 20L,
            TRUE ~ 20L
          ))
        data <- dplyr::select(data, -rowid)
        data <- dplyr::group_by(data, `::cut_group::`)
        data <- dplyr::summarize_all(data, ~ sum(is.na(.)) / dplyr::n())
        data <- dplyr::select(data, -1)
        data <- dplyr::mutate(data, col_num = which(col_names %in% x__))
        data <- dplyr::mutate(data, bin_num = 1:n_breaks)
        data <- dplyr::mutate(data, col_name = x__)
        data <- dplyr::rename(data, value = 1)
        data <- dplyr::select(data, col_name, col_num, bin_num, value)
        
        data
      }
    )
  
  missing_tbl <- dplyr::bind_rows(frequency_list)
  missing_tbl <- 
    dplyr::mutate(missing_tbl, value = ifelse(value == 0, NA_real_, value))
  missing_tbl <- 
    dplyr::mutate(
      missing_tbl,
      col_name = factor(col_name, levels = col_names)
    )
  
  missing_tbl
}

# Get a tibble of missing value proportions, by column, for a table
# - works across all supported data sources
# - returns 'tibble' of x columns rows, 3 columns
get_missing_by_column_tbl <- function(data) {
  
  col_names <- get_table_column_names(data = data)
  
  missing_by_column_list <-
    lapply(
      col_names,
      FUN = function(x__) {
        
        data <- dplyr::select(data, dplyr::one_of(x__))
        data <- dplyr::group_by(data)
        data <- 
          dplyr::summarize_all(
            data,
            ~ sum(ifelse(is.na(.), 1, 0), na.rm = TRUE) / dplyr::n()
          )
        data <- dplyr::collect(data)
        data <- dplyr::mutate(data, col_num = which(col_names %in% x__))
        data <- dplyr::mutate(data, col_name = x__)
        data <- dplyr::rename(data, value = 1)
        
        data
      })
  
  missing_by_column_tbl <- dplyr::bind_rows(missing_by_column_list)
  missing_by_column_tbl <- 
    dplyr::mutate(missing_by_column_tbl, value = round(value, 2))
  missing_by_column_tbl <-
    dplyr::mutate(
      missing_by_column_tbl,
      col_name = factor(col_name, levels = colnames(data))
    )
  
  missing_by_column_tbl
}

# Get a ggplot2 plot of missing values by column (with up to 20 bins),
# supported by the `frequency_tbl` and `missing_by_column_tbl` objects
# - works across all supported data sources
# - returns a ggplot object
get_missing_value_plot <- function(data, frequency_tbl, missing_by_column_tbl) {
  
  n_rows <- get_table_total_rows(data = data)
  
  ggplot2::ggplot(
    frequency_tbl,
    ggplot2::aes(x = col_name, y = bin_num, fill = value)
  ) +
    ggplot2::geom_tile(color = "white", linejoin = "bevel") +
    ggplot2::scale_fill_gradientn(
      colours = c("gray85", "black"),
      na.value = "#A1C1E5",
      limits = c(0, 1)
    ) +
    ggplot2::scale_y_continuous(
      breaks = c(0, 1, 20),
      labels = c("", as.character(n_rows), "1")
    ) +
    ggplot2::geom_label(
      data = missing_by_column_tbl,
      mapping = ggplot2::aes(
        x = col_name,
        y = -0.2,
        label = value,
        color = value
      ),
      fill = "white",
      show.legend = FALSE
    ) +
    ggplot2::labs(x = "", y = "") + 
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        angle = 90,
        vjust = 0.5,
        hjust = 1,
        size = 10,
        margin = ggplot2::margin(t = -1)
      ),
      axis.text.y = ggplot2::element_text(
        angle = 90,
        hjust = 0,
        margin = ggplot2::margin(r = -3)
      ),
      panel.grid = ggplot2::element_blank(),
      legend.direction = "horizontal",
      legend.title = ggplot2::element_blank(),
      legend.position = c(0.5, 1.0),
      plot.margin = ggplot2::unit(c(1, 0.5, 0, 0), "cm"),
      legend.key.width = ggplot2::unit(2.0, "cm"),
      legend.key.height = ggplot2::unit(3.0, "mm")
    )
}

get_table_slice_gt <- function(data_column,
                               locale) {
  
  data_column %>%
    gt::gt() %>%
    gt::fmt_percent(columns = 3, locale = locale) %>%
    gt::fmt_missing(columns = 1, missing_text = "**NA**") %>%
    gt::text_transform(
      locations = gt::cells_body(columns = 1),
      fn = function(x) ifelse(x == "**NA**", "<code>NA</code>", x)
    ) %>%
    gt::tab_options(
      table.border.top.style = "none",
      table.width = "100%"
    )
}
