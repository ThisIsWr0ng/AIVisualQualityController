namespace Aivqc.Trainer.Models;

public sealed record TrainingRunResult(
    string RunDirectory,
    string CheckpointPath,
    string OnnxPath,
    string MetricsPath,
    double Map50,
    double Map50To95,
    double Precision,
    double Recall,
    double F1);
