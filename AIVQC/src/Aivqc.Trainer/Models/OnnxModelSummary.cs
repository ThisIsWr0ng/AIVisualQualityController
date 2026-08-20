namespace Aivqc.Trainer.Models;

public sealed record OnnxModelSummary(
    string FileName,
    string SourcePath,
    long FileSizeBytes,
    string Sha256,
    IReadOnlyList<string> Inputs,
    IReadOnlyList<string> Outputs);
