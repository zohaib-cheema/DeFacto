// Invoke this function with the folder produced by defacto export.
(ExportFolder as text) as table =>
let
    Source = Csv.Document(
        File.Contents(ExportFolder & "/experiments.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {
        {"run_id", type text}, {"created_at", type text},
        {"model", type text}, {"mode", type text},
        {"train_rows", Int64.Type}, {"test_rows", Int64.Type},
        {"accuracy", type number}, {"balanced_accuracy", type number},
        {"macro_f1", type number}, {"majority_accuracy", type number},
        {"cv_macro_f1_mean", type number}
    }, "en-US")
in
    Typed
