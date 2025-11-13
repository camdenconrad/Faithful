using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// Trains a structural rule graph by analyzing grayscale structural properties of images
/// </summary>
public class StructureMapTrainer
{
    private const int DefaultStructureClasses = 32;

    /// <summary>
    /// Computes luminance from RGB: L = 0.299R + 0.587G + 0.114B
    /// </summary>
    private static double ComputeLuminance(byte r, byte g, byte b)
    {
        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    /// <summary>
    /// Computes gradient magnitude using Sobel operator
    /// </summary>
    private static double ComputeGradientMagnitude(double[,] luminanceMap, int x, int y, int width, int height)
    {
        // Sobel kernels
        int[,] sobelX = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        int[,] sobelY = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

        double gx = 0, gy = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    double lum = luminanceMap[nx, ny];
                    gx += lum * sobelX[dy + 1, dx + 1];
                    gy += lum * sobelY[dy + 1, dx + 1];
                }
            }
        }

        return Math.Sqrt(gx * gx + gy * gy);
    }

    /// <summary>
    /// Computes local entropy (diversity of neighboring intensities)
    /// </summary>
    private static double ComputeLocalEntropy(double[,] luminanceMap, int x, int y, int width, int height, int windowSize = 3)
    {
        var intensities = new List<double>();
        int halfWindow = windowSize / 2;

        for (int dy = -halfWindow; dy <= halfWindow; dy++)
        {
            for (int dx = -halfWindow; dx <= halfWindow; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    intensities.Add(luminanceMap[nx, ny]);
                }
            }
        }

        if (intensities.Count == 0) return 0.0;

        // Create histogram with 16 bins
        int numBins = 16;
        int[] histogram = new int[numBins];

        foreach (double intensity in intensities)
        {
            int bin = Math.Min((int)(intensity / 256.0 * numBins), numBins - 1);
            histogram[bin]++;
        }

        // Compute entropy: -sum(p * log2(p))
        double entropy = 0.0;
        int totalCount = intensities.Count;

        foreach (int count in histogram)
        {
            if (count > 0)
            {
                double p = (double)count / totalCount;
                entropy -= p * Math.Log2(p);
            }
        }

        return entropy;
    }

    /// <summary>
    /// Extracts structural metrics from an image
    /// </summary>
    private static StructuralMetrics[,] ExtractStructuralMetrics(Bitmap bitmap)
    {
        int width = bitmap.PixelSize.Width;
        int height = bitmap.PixelSize.Height;

        // First pass: compute luminance map
        double[,] luminanceMap = new double[width, height];

        // Convert to WriteableBitmap for pixel access
        var writeableBitmap = new WriteableBitmap(
            bitmap.PixelSize,
            new Avalonia.Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Unpremul);

        using (var ms = new System.IO.MemoryStream())
        {
            bitmap.Save(ms);
            ms.Position = 0;
            writeableBitmap = WriteableBitmap.Decode(ms);
        }

        unsafe
        {
            using var lockedBitmap = writeableBitmap.Lock();
            byte* ptr = (byte*)lockedBitmap.Address.ToPointer();
            int stride = lockedBitmap.RowBytes;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int offset = y * stride + x * 4;
                    byte b = ptr[offset];
                    byte g = ptr[offset + 1];
                    byte r = ptr[offset + 2];

                    luminanceMap[x, y] = ComputeLuminance(r, g, b);
                }
            }
        }

        // Second pass: compute gradients and entropy
        var metrics = new StructuralMetrics[width, height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double luminance = luminanceMap[x, y];
                double gradient = ComputeGradientMagnitude(luminanceMap, x, y, width, height);
                double entropy = ComputeLocalEntropy(luminanceMap, x, y, width, height);

                metrics[x, y] = new StructuralMetrics(luminance, gradient, entropy);
            }
        }

        return metrics;
    }

    /// <summary>
    /// Quantizes structural metrics into discrete structure classes using K-means
    /// </summary>
    private static void QuantizeStructures(StructuralMetrics[,] metrics, int numClasses)
    {
        int width = metrics.GetLength(0);
        int height = metrics.GetLength(1);

        // Collect all metrics
        var allMetrics = new List<StructuralMetrics>();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                allMetrics.Add(metrics[x, y]);
            }
        }

        // Initialize centroids randomly
        var random = new Random();
        var centroids = allMetrics.OrderBy(_ => random.Next()).Take(numClasses).ToList();

        // K-means clustering
        for (int iter = 0; iter < 10; iter++)
        {
            // Assignment step
            foreach (var metric in allMetrics)
            {
                double minDist = double.MaxValue;
                int bestClass = 0;

                for (int c = 0; c < centroids.Count; c++)
                {
                    double dist = metric.DistanceTo(centroids[c]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        bestClass = c;
                    }
                }

                metric.StructureClass = bestClass;
            }

            // Update step
            for (int c = 0; c < numClasses; c++)
            {
                var classMembers = allMetrics.Where(m => m.StructureClass == c).ToList();
                if (classMembers.Count > 0)
                {
                    double avgLum = classMembers.Average(m => m.Luminance);
                    double avgGrad = classMembers.Average(m => m.GradientMagnitude);
                    double avgEntropy = classMembers.Average(m => m.Entropy);

                    centroids[c] = new StructuralMetrics(avgLum, avgGrad, avgEntropy);
                }
            }
        }
    }

    /// <summary>
    /// Trains a structural rule graph from a collection of images
    /// </summary>
    public static async Task<StructuralRuleGraph> TrainAsync(
        List<Bitmap> images,
        Dictionary<uint, List<uint>> quantizedColorMap,
        int numStructureClasses = DefaultStructureClasses,
        IProgress<string>? progress = null)
    {
        var graph = new StructuralRuleGraph { NumStructureClasses = numStructureClasses };

        await Task.Run(() =>
        {
            progress?.Report("Extracting structural metrics from images...");

            for (int imgIdx = 0; imgIdx < images.Count; imgIdx++)
            {
                var bitmap = images[imgIdx];
                progress?.Report($"Processing image {imgIdx + 1}/{images.Count}...");

                // Extract structural metrics
                var metrics = ExtractStructuralMetrics(bitmap);

                // Quantize into structure classes
                QuantizeStructures(metrics, numStructureClasses);

                // Record color observations for each structure class
                int width = bitmap.PixelSize.Width;
                int height = bitmap.PixelSize.Height;

                // Convert to WriteableBitmap for pixel access
                WriteableBitmap writeableBitmap;
                using (var ms = new System.IO.MemoryStream())
                {
                    bitmap.Save(ms);
                    ms.Position = 0;
                    writeableBitmap = WriteableBitmap.Decode(ms);
                }

                unsafe
                {
                    using var lockedBitmap = writeableBitmap.Lock();
                    byte* ptr = (byte*)lockedBitmap.Address.ToPointer();
                    int stride = lockedBitmap.RowBytes;

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int offset = y * stride + x * 4;
                            byte b = ptr[offset];
                            byte g = ptr[offset + 1];
                            byte r = ptr[offset + 2];
                            byte a = ptr[offset + 3];

                            uint originalColor = (uint)((a << 24) | (r << 16) | (g << 8) | b);
                            uint quantizedColor = GetQuantizedColor(originalColor, quantizedColorMap);

                            int structureClass = metrics[x, y].StructureClass;
                            graph.RecordObservation(structureClass, quantizedColor);
                        }
                    }
                }
            }

            progress?.Report("Normalizing structural rule graph...");
            graph.Normalize();

            progress?.Report("Structural rule graph training complete!");
        });

        return graph;
    }

    /// <summary>
    /// Finds the quantized color for an original color
    /// </summary>
    private static uint GetQuantizedColor(uint originalColor, Dictionary<uint, List<uint>> quantizedColorMap)
    {
        foreach (var (quantized, originals) in quantizedColorMap)
        {
            if (originals.Contains(originalColor))
            {
                return quantized;
            }
        }
        return originalColor; // Fallback
    }
}
