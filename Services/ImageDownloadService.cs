using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace NNImage.Services;

public class ImageDownloadService : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;
    private readonly List<string> _imageApis;
    private int _currentApiIndex = 0;
    private readonly Random _random = new();

    public ImageDownloadService()
    {
        _httpClient = new HttpClient();
        _httpClient.Timeout = TimeSpan.FromSeconds(30);

        // Create cache directory for downloaded images
        _cacheDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "NNImage",
            "TrainingCache");

        Directory.CreateDirectory(_cacheDirectory);

        // List of free image APIs (no API key needed)
        _imageApis = new List<string>
        {
            "https://picsum.photos/512/512", // Lorem Picsum - completely free
            "https://source.unsplash.com/random/512x512", // Unsplash random
            "https://loremflickr.com/512/512", // LoremFlickr
        };

        Console.WriteLine($"[ImageDownload] Initialized with {_imageApis.Count} image sources");
        Console.WriteLine($"[ImageDownload] Cache directory: {_cacheDirectory}");
    }

    public async Task<string?> DownloadRandomImageAsync(CancellationToken cancellationToken = default)
    {
        for (int attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                // Rotate through APIs to avoid rate limiting
                var apiUrl = _imageApis[_currentApiIndex];
                _currentApiIndex = (_currentApiIndex + 1) % _imageApis.Count;

                // Add random query to get different images
                var randomSeed = _random.Next(0, 100000);
                var url = $"{apiUrl}?random={randomSeed}";

                Console.WriteLine($"[ImageDownload] Downloading from {apiUrl}...");

                var response = await _httpClient.GetAsync(url, cancellationToken);
                response.EnsureSuccessStatusCode();

                var imageData = await response.Content.ReadAsByteArrayAsync(cancellationToken);

                // Save to cache with timestamp
                var fileName = $"train_{DateTime.Now:yyyyMMdd_HHmmss}_{randomSeed}.jpg";
                var filePath = Path.Combine(_cacheDirectory, fileName);

                await File.WriteAllBytesAsync(filePath, imageData, cancellationToken);

                Console.WriteLine($"[ImageDownload] Downloaded image: {fileName} ({imageData.Length / 1024} KB)");
                return filePath;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ImageDownload] Download attempt {attempt + 1} failed: {ex.Message}");

                if (cancellationToken.IsCancellationRequested)
                    break;

                // Wait before retry with exponential backoff
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), cancellationToken);
            }
        }

        return null;
    }

    public async Task<List<string>> DownloadBatchAsync(int count, CancellationToken cancellationToken = default)
    {
        var downloadedFiles = new List<string>();
        var tasks = new List<Task<string?>>();

        Console.WriteLine($"[ImageDownload] Starting batch download of {count} images...");

        for (int i = 0; i < count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            tasks.Add(DownloadRandomImageAsync(cancellationToken));

            // Add delay to avoid hammering servers
            await Task.Delay(500, cancellationToken);
        }

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
            if (result != null)
                downloadedFiles.Add(result);
        }

        Console.WriteLine($"[ImageDownload] Batch complete: {downloadedFiles.Count}/{count} images downloaded");
        return downloadedFiles;
    }

    public void ClearCache()
    {
        try
        {
            var files = Directory.GetFiles(_cacheDirectory);
            Console.WriteLine($"[ImageDownload] Clearing {files.Length} cached images...");

            foreach (var file in files)
            {
                try
                {
                    File.Delete(file);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ImageDownload] Failed to delete {file}: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ImageDownload] Cache clear failed: {ex.Message}");
        }
    }

    public long GetCacheSize()
    {
        try
        {
            var files = Directory.GetFiles(_cacheDirectory);
            return files.Sum(f => new FileInfo(f).Length);
        }
        catch
        {
            return 0;
        }
    }

    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}
