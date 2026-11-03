PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    train_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, metric)
);
CREATE TABLE IF NOT EXISTS confusion (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    actual TEXT NOT NULL,
    predicted TEXT NOT NULL,
    count INTEGER NOT NULL CHECK(count >= 0),
    PRIMARY KEY (run_id, actual, predicted)
);
CREATE VIEW IF NOT EXISTS experiment_summary AS
SELECT r.run_id, r.created_at, r.model, r.mode, r.train_rows, r.test_rows,
       MAX(CASE WHEN m.metric = 'accuracy' THEN m.value END) AS accuracy,
       MAX(CASE WHEN m.metric = 'balanced_accuracy' THEN m.value END) AS balanced_accuracy,
       MAX(CASE WHEN m.metric = 'macro_f1' THEN m.value END) AS macro_f1,
       MAX(CASE WHEN m.metric = 'majority_accuracy' THEN m.value END) AS majority_accuracy,
       MAX(CASE WHEN m.metric = 'cv_macro_f1_mean' THEN m.value END) AS cv_macro_f1_mean
FROM runs r JOIN metrics m USING (run_id)
GROUP BY r.run_id;
