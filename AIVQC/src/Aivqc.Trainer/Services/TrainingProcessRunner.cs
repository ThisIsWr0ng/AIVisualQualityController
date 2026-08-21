using System.Diagnostics;
using System.Text.Json;
using Aivqc.Core.Training;
using Aivqc.Trainer.Models;

namespace Aivqc.Trainer.Services;

public sealed class TrainingProcessRunner
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public async Task<TrainingRunResult> RunAsync(
        TrainingConfiguration configuration,
        IProgress<TrainingJobEvent>? progress = null,
        CancellationToken cancellationToken = default)
    {
        configuration.Validate();
        Directory.CreateDirectory(configuration.OutputRoot);

        var temporaryDirectory = Path.Combine(Path.GetTempPath(), "AIVQC", "TrainingJobs");
        Directory.CreateDirectory(temporaryDirectory);
        var configurationPath = Path.Combine(temporaryDirectory, $"{Guid.NewGuid():N}.json");

        try
        {
            await WriteConfigurationAsync(configurationPath, configuration, cancellationToken);
            return await RunProcessAsync(configuration, configurationPath, progress, cancellationToken);
        }
        finally
        {
            File.Delete(configurationPath);
        }
    }

    private static async Task WriteConfigurationAsync(
        string path,
        TrainingConfiguration configuration,
        CancellationToken cancellationToken)
    {
        var payload = new
        {
            dataset_root = Path.GetFullPath(configuration.DatasetRoot),
            output_root = Path.GetFullPath(configuration.OutputRoot),
            run_name = configuration.RunName,
            epochs = configuration.Epochs,
            batch_size = configuration.BatchSize,
            learning_rate = configuration.LearningRate,
            workers = configuration.Workers,
            device = configuration.Device,
            pretrained_backbone = configuration.PretrainedBackbone,
            score_threshold = configuration.ScoreThreshold,
            seed = 42,
            max_samples = configuration.MaxSamples,
        };
        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, payload, JsonOptions, cancellationToken);
    }

    private static async Task<TrainingRunResult> RunProcessAsync(
        TrainingConfiguration configuration,
        string configurationPath,
        IProgress<TrainingJobEvent>? progress,
        CancellationToken cancellationToken)
    {
        using var process = new Process
        {
            StartInfo = CreateStartInfo(configuration, configurationPath),
            EnableRaisingEvents = true,
        };

        if (!process.Start())
        {
            throw new InvalidOperationException("The Python training process could not be started.");
        }

        using var cancellationRegistration = cancellationToken.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (InvalidOperationException)
            {
                // The process completed between the state check and termination request.
            }
        });

        TrainingJobEvent? completedEvent = null;
        TrainingJobEvent? failedEvent = null;
        var standardErrorTask = process.StandardError.ReadToEndAsync();

        while (await process.StandardOutput.ReadLineAsync() is { } line)
        {
            var trainingEvent = TrainingEventJson.Parse(line);
            if (trainingEvent is null)
            {
                continue;
            }

            progress?.Report(trainingEvent);
            completedEvent = trainingEvent.Type == "completed" ? trainingEvent : completedEvent;
            failedEvent = trainingEvent.Type == "failed" ? trainingEvent : failedEvent;
        }

        await process.WaitForExitAsync(CancellationToken.None);
        var standardError = await standardErrorTask;
        cancellationToken.ThrowIfCancellationRequested();

        if (process.ExitCode != 0 || completedEvent is null)
        {
            var message = failedEvent?.Message;
            if (string.IsNullOrWhiteSpace(message))
            {
                message = string.IsNullOrWhiteSpace(standardError)
                    ? $"Training exited with code {process.ExitCode}."
                    : standardError.Trim();
            }

            throw new InvalidOperationException(message);
        }

        return new TrainingRunResult(
            completedEvent.RunDirectory ?? Path.GetDirectoryName(completedEvent.OnnxPath) ?? string.Empty,
            completedEvent.CheckpointPath ?? string.Empty,
            completedEvent.OnnxPath ?? string.Empty,
            completedEvent.MetricsPath ?? string.Empty,
            completedEvent.Map50 ?? 0,
            completedEvent.Map50To95 ?? 0,
            completedEvent.Precision ?? 0,
            completedEvent.Recall ?? 0,
            completedEvent.F1 ?? 0);
    }

    private static ProcessStartInfo CreateStartInfo(
        TrainingConfiguration configuration,
        string configurationPath)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = configuration.PythonExecutable,
            WorkingDirectory = Path.GetDirectoryName(configuration.BackendScriptPath)!,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        startInfo.ArgumentList.Add(configuration.BackendScriptPath);
        startInfo.ArgumentList.Add("--config");
        startInfo.ArgumentList.Add(configurationPath);
        return startInfo;
    }
}
