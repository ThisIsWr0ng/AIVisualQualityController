namespace Aivqc.Trainer.Models;

public sealed record OnnxModelSummary(
    string FileName,
    string SourcePath,
    long FileSizeBytes,
    string Sha256,
    int InputWidth,
    int InputHeight,
    IReadOnlyDictionary<string, int> ClassNames,
    IReadOnlyList<string> Inputs,
    IReadOnlyList<string> Outputs);
