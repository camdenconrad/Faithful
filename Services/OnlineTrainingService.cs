using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace NNImage.Services;

public class OnlineTrainingService : IDisposable
{
    private readonly ImageDownloadService _downloadService;
    private readonly GpuAccelerator _gpu;
    private CancellationTokenSource? _cancellationTokenSource;
    private Task? _trainingTask;
    private bool _isRunning;
    private int _totalImagesProcessed;
    private int _currentBatchSize = 5;
    private int _quantizationLevel = 128;

    public bool IsRunning => _isRunning;
    public int TotalImagesProcessed => _totalImagesProcessed;
    public int CurrentBatchSize => _currentBatchSize;

    public event Action<int, string>? ProgressUpdate; // (imagesProcessed, message)
    public event Action<Exception>? ErrorOccurred;

    public OnlineTrainingService(GpuAccelerator gpu)
    {
        _downloadService = new ImageDownloadService();
        _gpu = gpu;
        Console.WriteLine("[OnlineTraining] Service initialized");
    }

    public void Start(int batchSize = 5, int quantizationLevel = 128)
    {
        if (_isRunning)
        {
            Console.WriteLine("[OnlineTraining] Already running");
            return;
        }

        _currentBatchSize = batchSize;
        _quantizationLevel = quantizationLevel;
        _cancellationTokenSource = new CancellationTokenSource();
        _isRunning = true;

        Console.WriteLine($"[OnlineTraining] Starting with batch size: {batchSize}, quantization: {quantizationLevel}");

        _trainingTask = Task.Run(async () => await TrainingLoopAsync(_cancellationTokenSource.Token));
    }

    public void Stop()
    {
        if (!_isRunning)
            return;

        Console.WriteLine("[OnlineTraining] Stopping...");
        _cancellationTokenSource?.Cancel();
        _isRunning = false;
    }

    public async Task WaitForStopAsync()
    {
        if (_trainingTask != null)
        {
            try
            {
                await _trainingTask;
            }
            catch (OperationCanceledException)
            {
                // Expected when stopping
            }
        }
    }

    private async Task TrainingLoopAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Download batch of images
                    ProgressUpdate?.Invoke(_totalImagesProcessed, "Downloading images...");
                    var imagePaths = await _downloadService.DownloadBatchAsync(_currentBatchSize, cancellationToken);

                    if (imagePaths.Count == 0)
                    {
                        ProgressUpdate?.Invoke(_totalImagesProcessed, "No images downloaded, waiting...");
                        await Task.Delay(5000, cancellationToken);
                        continue;
                    }

                    // Extract pixel data on a background thread (not UI thread)
                    ProgressUpdate?.Invoke(_totalImagesProcessed, $"Processing {imagePaths.Count} images...");

                    var batchData = new List<(uint[] pixels, int width, int height)>();

                    foreach (var imagePath in imagePaths)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        try
                        {
                            var pixelData = await ExtractPixelDataAsync(imagePath, cancellationToken);
                            if (pixelData.HasValue)
                            {
                                batchData.Add(pixelData.Value);
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[OnlineTraining] Failed to process {imagePath}: {ex.Message}");
                        }
                    }

                    if (batchData.Count == 0)
                        continue;

                    // Train on this batch - this will be called by MainWindow to add to existing graph
                    ProgressUpdate?.Invoke(_totalImagesProcessed, $"Training on {batchData.Count} images...");

                    // This needs to be handled by the caller to add to existing graph
                    // For now, just signal that we have data ready
                    _totalImagesProcessed += batchData.Count;

                    ProgressUpdate?.Invoke(_totalImagesProcessed, $"✓ Processed batch of {batchData.Count} images");

                    // Wait between batches to avoid overwhelming the system
                    await Task.Delay(2000, cancellationToken);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    Console.WriteLine($"[OnlineTraining] Batch processing error: {ex.Message}");
                    ErrorOccurred?.Invoke(ex);

                    // Wait before retrying
                    await Task.Delay(5000, cancellationToken);
                }
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("[OnlineTraining] Training stopped by user");
        }
        finally
        {
            _isRunning = false;
            Console.WriteLine($"[OnlineTraining] Stopped. Total images processed: {_totalImagesProcessed}");
        }
    }

    private async Task<(uint[] pixels, int width, int height)?> ExtractPixelDataAsync(string imagePath, CancellationToken cancellationToken)
    {
        // This needs to be done carefully since Avalonia bitmaps need UI thread
        // For now, return null and let the caller handle it properly
        // This would need to be integrated with the MainWindow's extraction logic
        return null;
    }

    public void ClearCache()
    {
        _downloadService.ClearCache();
    }

    public long GetCacheSize()
    {
        return _downloadService.GetCacheSize();
    }

    public void Dispose()
    {
        Stop();
        _downloadService?.Dispose();
        _cancellationTokenSource?.Dispose();
    }
}
