namespace Aivqc.Trainer.Models;

public sealed record TrainingConfiguration(
    string PythonExecutable,
    string BackendScriptPath,
    string DatasetRoot,
    string OutputRoot,
    string RunName,
    int Epochs,
    int BatchSize,
    double LearningRate,
    int Workers,
    string Device,
    bool PretrainedBackbone,
    double ScoreThreshold,
    int MaxSamples = 0)
{
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(PythonExecutable))
        {
            throw new ArgumentException("Select a Python executable.");
        }

        if (!File.Exists(BackendScriptPath))
        {
            throw new FileNotFoundException("The AIVQC training backend is missing.", BackendScriptPath);
        }

        if (!Directory.Exists(DatasetRoot))
        {
            throw new DirectoryNotFoundException($"Dataset directory not found: {DatasetRoot}");
        }

        if (Epochs is < 1 or > 1000)
        {
            throw new ArgumentOutOfRangeException(nameof(Epochs), "Epochs must be between 1 and 1000.");
        }

        if (BatchSize is < 2 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(BatchSize), "Batch size must be between 2 and 256.");
        }

        if (LearningRate is <= 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(LearningRate), "Learning rate must be greater than 0 and at most 1.");
        }

        if (MaxSamples < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxSamples), "Max samples cannot be negative.");
        }
    }
}
