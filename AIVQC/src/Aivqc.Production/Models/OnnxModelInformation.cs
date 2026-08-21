namespace Aivqc.Production.Models;

public sealed record OnnxModelInformation(
    string FileName,
    string FilePath,
    int InputWidth,
    int InputHeight,
    IReadOnlyDictionary<int, string> ClassNames);
