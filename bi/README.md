# Reporting integration

Run `defacto export` to produce `experiments.csv`, `metrics.csv`, and `confusion.csv` from the recorded SQL store. Their join key is `run_id`.

1. In Power BI, create a blank query, paste `experiments.pq` into the advanced editor, and invoke the function with the export folder. Name the resulting table `experiments`. The companion DAX definitions select a single run so scores are not averaged across unrelated models.
2. In Tableau, connect to the CSV files and relate them on `run_id`. Use a single run filter. A confusion matrix places `actual` on rows, `predicted` on columns, and sums `count` as the cell value.
3. Use the generated HTML report to inspect the same measured scores without either desktop application.

The CSV exports and HTML report are exercised by automated tests. The Power Query and DAX sources are integration starting points; a native PBIX or Tableau workbook is not included or claimed as tested.
