using Avalonia.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace NNImage;

public class WaveFunctionCollapse
{
    private readonly AdjacencyGraph _adjacencyGraph;
    private readonly int _width;
    private readonly int _height;
    private readonly Random _random = new();
    private readonly HashSet<ColorRgb>?[,] _possibilities;
    private readonly ColorRgb?[,] _collapsed;

    public WaveFunctionCollapse(AdjacencyGraph adjacencyGraph, int width, int height)
    {
        _adjacencyGraph = adjacencyGraph;
        _width = width;
        _height = height;
        _possibilities = new HashSet<ColorRgb>?[height, width];
        _collapsed = new ColorRgb?[height, width];

        InitializePossibilities();
    }

    private void InitializePossibilities()
    {
        var allColors = _adjacencyGraph.GetAllColors();

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                _possibilities[y, x] = new HashSet<ColorRgb>(allColors);
            }
        }
    }

    public Bitmap Generate()
    {
        int maxIterations = _width * _height * 2;
        int iteration = 0;

        while (iteration < maxIterations)
        {
            // Find cell with minimum entropy
            var (x, y) = FindMinimumEntropyCell();

            if (x == -1)
                break; // All cells collapsed

            // Collapse the cell
            CollapseCell(x, y);

            // Propagate constraints
            Propagate(x, y);

            iteration++;
        }

        // Fill any remaining uncollapsed cells
        FillRemainingCells();

        return CreateBitmap();
    }

    private (int x, int y) FindMinimumEntropyCell()
    {
        int minEntropy = int.MaxValue;
        var candidates = new List<(int x, int y)>();

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_collapsed[y, x] != null)
                    continue;

                var entropy = _possibilities[y, x]?.Count ?? 0;

                if (entropy == 0)
                    continue;

                if (entropy < minEntropy)
                {
                    minEntropy = entropy;
                    candidates.Clear();
                    candidates.Add((x, y));
                }
                else if (entropy == minEntropy)
                {
                    candidates.Add((x, y));
                }
            }
        }

        if (candidates.Count == 0)
            return (-1, -1);

        return candidates[_random.Next(candidates.Count)];
    }

    private void CollapseCell(int x, int y)
    {
        var possibleColors = _possibilities[y, x];
        if (possibleColors == null || possibleColors.Count == 0)
        {
            // Fallback to random color
            var allColors = _adjacencyGraph.GetAllColors();
            _collapsed[y, x] = allColors[_random.Next(allColors.Count)];
            return;
        }

        // Weight selection by neighbor compatibility
        var weights = new Dictionary<ColorRgb, double>();

        foreach (var color in possibleColors)
        {
            double weight = 1.0;

            // Check compatibility with already collapsed neighbors
            foreach (var direction in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = direction.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var neighborColor = _collapsed[ny, nx];
                    if (neighborColor != null)
                    {
                        var possibleNeighbors = _adjacencyGraph.GetPossibleNeighbors(color, direction);
                        if (possibleNeighbors.Contains(neighborColor.Value))
                            weight *= 2.0; // Increase weight for compatible colors
                    }
                }
            }

            weights[color] = weight;
        }

        // Weighted random selection
        var totalWeight = weights.Values.Sum();
        var rand = _random.NextDouble() * totalWeight;
        var cumulative = 0.0;

        foreach (var (color, weight) in weights)
        {
            cumulative += weight;
            if (rand <= cumulative)
            {
                _collapsed[y, x] = color;
                _possibilities[y, x] = null;
                return;
            }
        }

        // Fallback
        _collapsed[y, x] = possibleColors.First();
        _possibilities[y, x] = null;
    }

    private void Propagate(int startX, int startY)
    {
        var queue = new Queue<(int x, int y)>();
        queue.Enqueue((startX, startY));

        while (queue.Count > 0)
        {
            var (x, y) = queue.Dequeue();
            var currentColor = _collapsed[y, x];

            if (currentColor == null)
                continue;

            foreach (var direction in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = direction.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx < 0 || nx >= _width || ny < 0 || ny >= _height)
                    continue;

                if (_collapsed[ny, nx] != null)
                    continue;

                var neighborPossibilities = _possibilities[ny, nx];
                if (neighborPossibilities == null || neighborPossibilities.Count == 0)
                    continue;

                // Get valid neighbors from adjacency graph
                var validNeighbors = _adjacencyGraph.GetPossibleNeighbors(currentColor.Value, direction);

                // Intersect with current possibilities
                var newPossibilities = neighborPossibilities.Intersect(validNeighbors).ToHashSet();

                if (newPossibilities.Count < neighborPossibilities.Count)
                {
                    _possibilities[ny, nx] = newPossibilities.Count > 0 ? newPossibilities : neighborPossibilities;

                    if (newPossibilities.Count > 0)
                        queue.Enqueue((nx, ny));
                }
            }
        }
    }

    private void FillRemainingCells()
    {
        var allColors = _adjacencyGraph.GetAllColors();

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_collapsed[y, x] == null)
                {
                    var possibilities = _possibilities[y, x];
                    if (possibilities != null && possibilities.Count > 0)
                        _collapsed[y, x] = possibilities.ElementAt(_random.Next(possibilities.Count));
                    else
                        _collapsed[y, x] = allColors[_random.Next(allColors.Count)];
                }
            }
        }
    }

    private Bitmap CreateBitmap()
    {
        var bitmap = new WriteableBitmap(
            new Avalonia.PixelSize(_width, _height),
            new Avalonia.Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Opaque);

        using var lockedBitmap = bitmap.Lock();

        unsafe
        {
            var ptr = (uint*)lockedBitmap.Address.ToPointer();

            for (int y = 0; y < _height; y++)
            {
                for (int x = 0; x < _width; x++)
                {
                    var color = _collapsed[y, x] ?? new ColorRgb(0, 0, 0);
                    ptr[y * _width + x] = ColorToPixel(color);
                }
            }
        }

        return bitmap;
    }

    private uint ColorToPixel(ColorRgb color)
    {
        return (uint)((255 << 24) | (color.R << 16) | (color.G << 8) | color.B);
    }
}
