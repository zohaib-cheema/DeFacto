SELECT actual,
       SUM(count) AS evaluated_rows,
       SUM(CASE WHEN actual <> predicted THEN count ELSE 0 END) AS errors,
       ROUND(1.0 * SUM(CASE WHEN actual = predicted THEN count ELSE 0 END)
             / NULLIF(SUM(count), 0), 4) AS recall
FROM confusion
WHERE run_id = ?
GROUP BY actual
ORDER BY actual;
