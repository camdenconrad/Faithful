using Avalonia.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// Ultra-fast Wave Function Collapse using node-based graph with multiple random seeds
/// Grows image from random starting points with real-time visualization
/// </summary>
public class FastWaveFunctionCollapse
{
    private readonly FastContextGraph _graph;
    private readonly int _width;
    private readonly int _height;
    private readonly Random _random = new();
    private readonly ColorRgb?[,] _collapsed;
    private readonly List<ColorRgb> _availableColors;

    public delegate void ProgressCallback(uint[] pixels, int collapsedCount, int totalCells);

    public FastWaveFunctionCollapse(FastContextGraph graph, int width, int height)
    {
        _graph = graph;
        _width = width;
        _height = height;
        _collapsed = new ColorRgb?[height, width];
        _availableColors = graph.GetAllColors();

        if (_availableColors.Count == 0)
        {
            throw new InvalidOperationException("Graph has no colors - train the model first!");
        }

        Console.WriteLine($"[FastWFC] Initialized for {width}x{height} with {_availableColors.Count} colors");
    }

    /// <summary>
    /// Generate image using multi-seed WFC with real-time growth visualization
    /// </summary>
    public uint[] Generate(int seedCount = 5, ProgressCallback? progressCallback = null, int updateFrequency = 10)
    {
        Console.WriteLine($"[FastWFC] Starting generation with {seedCount} random seeds");

        // Initialize random seed points
        var seeds = InitializeSeeds(seedCount);
        Console.WriteLine($"[FastWFC] Placed {seeds.Count} seeds");

        // Growth wave queue - cells to process next
        var waveQueue = new Queue<(int x, int y, int generation)>();
        var processed = new HashSet<(int x, int y)>();

        // Add seeds to wave queue
        foreach (var (x, y) in seeds)
        {
            waveQueue.Enqueue((x, y, 0));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Grow outward from seeds in waves
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            var (x, y, generation) = waveQueue.Dequeue();

            // Collapse current cell if not already collapsed
            if (_collapsed[y, x] == null)
            {
                CollapseCell(x, y);
                collapsedCount++;
            }

            // Add neighboring cells to wave (they'll grow next)
            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        waveQueue.Enqueue((nx, ny, generation + 1));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;

            // Real-time progress update
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        Console.WriteLine($"[FastWFC] Wave propagation complete - filling remaining cells");

        // Fill any remaining uncollapsed cells (rare)
        FillRemainingCells();

        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] Generation complete!");
        return finalPixels;
    }

    /// <summary>
    /// Initialize random seed points across the image
    /// </summary>
    private List<(int x, int y)> InitializeSeeds(int seedCount)
    {
        var seeds = new List<(int x, int y)>();

        // Distribute seeds randomly but avoid clustering
        var minDistance = Math.Min(_width, _height) / (int)Math.Sqrt(seedCount);

        for (int i = 0; i < seedCount * 3 && seeds.Count < seedCount; i++)
        {
            var x = _random.Next(_width);
            var y = _random.Next(_height);

            // Check distance from existing seeds
            bool tooClose = false;
            foreach (var (sx, sy) in seeds)
            {
                var dist = Math.Sqrt((x - sx) * (x - sx) + (y - sy) * (y - sy));
                if (dist < minDistance)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
            {
                // Pick random color for this seed
                var seedColor = _availableColors[_random.Next(_availableColors.Count)];
                _collapsed[y, x] = seedColor;
                seeds.Add((x, y));
                Console.WriteLine($"[FastWFC] Seed {seeds.Count} at ({x}, {y}) = {seedColor}");
            }
        }

        // Fallback: if we couldn't place enough seeds, just place them randomly
        while (seeds.Count < seedCount)
        {
            var x = _random.Next(_width);
            var y = _random.Next(_height);
            if (_collapsed[y, x] == null)
            {
                var seedColor = _availableColors[_random.Next(_availableColors.Count)];
                _collapsed[y, x] = seedColor;
                seeds.Add((x, y));
            }
        }

        return seeds;
    }

    /// <summary>
    /// Collapse a cell using weighted predictions from nearby nodes
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CollapseCell(int x, int y)
    {
        // Calculate normalized position
        var normalizedX = _width > 1 ? (float)x / (_width - 1) : 0.5f;
        var normalizedY = _height > 1 ? (float)y / (_height - 1) : 0.5f;

        // Collect weighted predictions from all collapsed neighbors
        var colorWeights = new Dictionary<ColorRgb, double>();

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            var (dx, dy) = dir.GetOffset();
            var nx = x + dx;
            var ny = y + dy;

            if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
            {
                var neighborColor = _collapsed[ny, nx];
                if (neighborColor.HasValue)
                {
                    // Get predictions from graph based on neighbor's color and position
                    var neighborNormX = _width > 1 ? (float)nx / (_width - 1) : 0.5f;
                    var neighborNormY = _height > 1 ? (float)ny / (_height - 1) : 0.5f;

                    var predictions = _graph.GetWeightedNeighbors(
                        neighborColor.Value, 
                        neighborNormX, 
                        neighborNormY, 
                        dir);

                    // Accumulate weights from this neighbor's predictions
                    foreach (var (color, weight) in predictions)
                    {
                        if (!colorWeights.ContainsKey(color))
                        {
                            colorWeights[color] = 0;
                        }
                        colorWeights[color] += weight;
                    }
                }
            }
        }

        // Select color based on accumulated weights
        ColorRgb selectedColor;

        if (colorWeights.Count > 0)
        {
            // Weighted random selection based on edge weights
            var totalWeight = colorWeights.Values.Sum();
            var rand = _random.NextDouble() * totalWeight;
            var cumulative = 0.0;

            selectedColor = colorWeights.First().Key; // Fallback

            foreach (var (color, weight) in colorWeights.OrderByDescending(kvp => kvp.Value))
            {
                cumulative += weight;
                if (rand <= cumulative)
                {
                    selectedColor = color;
                    break;
                }
            }
        }
        else
        {
            // No neighbors collapsed yet - pick random color
            selectedColor = _availableColors[_random.Next(_availableColors.Count)];
        }

        _collapsed[y, x] = selectedColor;
    }

    private void FillRemainingCells()
    {
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_collapsed[y, x] == null)
                {
                    // Try to collapse based on neighbors
                    CollapseCell(x, y);
                }
            }
        }
    }

    private uint[] CreatePixelData()
    {
        var pixels = new uint[_width * _height];

        System.Threading.Tasks.Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                var color = _collapsed[y, x] ?? new ColorRgb(32, 32, 32); // Dark gray for uncollapsed
                pixels[y * _width + x] = ColorToPixel(color);
            }
        });

        return pixels;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint ColorToPixel(ColorRgb color)
    {
        return (uint)((255 << 24) | (color.R << 16) | (color.G << 8) | color.B);
    }

    public static Bitmap CreateBitmapFromPixels(uint[] pixels, int width, int height)
    {
        var bitmap = new WriteableBitmap(
            new Avalonia.PixelSize(width, height),
            new Avalonia.Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Opaque);

        using var lockedBitmap = bitmap.Lock();

        unsafe
        {
            var ptr = (uint*)lockedBitmap.Address.ToPointer();
            for (int i = 0; i < width * height; i++)
            {
                ptr[i] = pixels[i];
            }
        }

        return bitmap;
    }
}
