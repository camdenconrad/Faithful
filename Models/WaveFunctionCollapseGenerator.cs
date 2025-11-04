using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NNImage.Models;

/// <summary>
/// Wave Function Collapse generator that builds images from random seeds
/// Shows real-time growth as patterns propagate outward
/// </summary>
public class WaveFunctionCollapseGenerator
{
    private readonly FastContextGraph _graph;
    private readonly Random _random;

    // Generation grid
    private ColorRgb?[,] _grid;
    private HashSet<ColorRgb>[,] _possibleColors; // Superposition state for each cell
    private int _width;
    private int _height;

    // Real-time callback for visualization
    public Action<int, int, ColorRgb>? OnPixelCollapsed;
    public Action<string>? OnStatusUpdate;

    // Generation statistics
    private int _collapsedCells = 0;
    private int _totalCells = 0;

    public WaveFunctionCollapseGenerator(FastContextGraph graph, int? seed = null)
    {
        _graph = graph;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generate image using Wave Function Collapse from random seed points
    /// </summary>
    public async Task<ColorRgb[,]> GenerateAsync(
        int width, 
        int height, 
        int seedCount = 3,
        int delayMs = 1,
        CancellationToken cancellationToken = default)
    {
        _width = width;
        _height = height;
        _totalCells = width * height;
        _collapsedCells = 0;

        // Initialize grid
        _grid = new ColorRgb?[height, width];
        _possibleColors = new HashSet<ColorRgb>[height, width];

        var allColors = _graph.GetAllColors();
        if (allColors.Count == 0)
        {
            throw new InvalidOperationException("Graph has no colors! Train the model first.");
        }

        OnStatusUpdate?.Invoke($"Initializing {width}x{height} grid with {allColors.Count} possible colors...");

        // Initialize all cells with all possible colors (superposition)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                _possibleColors[y, x] = new HashSet<ColorRgb>(allColors);
            }
        }

        // Place random seed points
        OnStatusUpdate?.Invoke($"Placing {seedCount} random seed points...");
        await PlaceRandomSeeds(seedCount, delayMs, cancellationToken);

        // Wave Function Collapse main loop
        OnStatusUpdate?.Invoke("Starting Wave Function Collapse...");

        while (_collapsedCells < _totalCells && !cancellationToken.IsCancellationRequested)
        {
            // Find cell with lowest entropy (most constrained)
            var (x, y) = FindLowestEntropyCell();

            if (x == -1 || y == -1)
            {
                // No uncollapsed cells or contradiction - fill remaining randomly
                OnStatusUpdate?.Invoke("Filling remaining cells...");
                await FillRemaining(delayMs, cancellationToken);
                break;
            }

            // Collapse the cell
            await CollapseCell(x, y, delayMs, cancellationToken);

            // Propagate constraints to neighbors
            PropagateConstraints(x, y);

            // Status update every 100 cells
            if (_collapsedCells % 100 == 0)
            {
                var progress = (_collapsedCells * 100.0) / _totalCells;
                OnStatusUpdate?.Invoke($"Generating: {progress:F1}% ({_collapsedCells}/{_totalCells} cells)");
            }
        }

        OnStatusUpdate?.Invoke($"Generation complete! {_collapsedCells} cells collapsed.");

        // Convert to final grid
        var result = new ColorRgb[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                result[y, x] = _grid[y, x] ?? allColors[_random.Next(allColors.Count)];
            }
        }

        return result;
    }

    private async Task PlaceRandomSeeds(int count, int delayMs, CancellationToken cancellationToken)
    {
        var allColors = _graph.GetAllColors();

        for (int i = 0; i < count && !cancellationToken.IsCancellationRequested; i++)
        {
            var x = _random.Next(_width);
            var y = _random.Next(_height);

            // Choose random color weighted by observation count
            var color = allColors[_random.Next(allColors.Count)];

            // Collapse seed cell
            _grid[y, x] = color;
            _possibleColors[y, x].Clear();
            _possibleColors[y, x].Add(color);
            _collapsedCells++;

            OnPixelCollapsed?.Invoke(x, y, color);

            if (delayMs > 0)
            {
                await Task.Delay(delayMs, cancellationToken);
            }
        }
    }

    private (int x, int y) FindLowestEntropyCell()
    {
        int minEntropy = int.MaxValue;
        var candidates = new List<(int x, int y)>();

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_grid[y, x].HasValue) continue; // Already collapsed

                var entropy = _possibleColors[y, x].Count;

                if (entropy == 0) continue; // Contradiction - skip

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
        {
            return (-1, -1);
        }

        // Randomly pick from candidates with same entropy (breaks ties randomly)
        return candidates[_random.Next(candidates.Count)];
    }

    private async Task CollapseCell(int x, int y, int delayMs, CancellationToken cancellationToken)
    {
        var possible = _possibleColors[y, x];

        if (possible.Count == 0)
        {
            // Contradiction - choose random color
            var allColors = _graph.GetAllColors();
            var color = allColors[_random.Next(allColors.Count)];
            _grid[y, x] = color;
            _collapsedCells++;
            OnPixelCollapsed?.Invoke(x, y, color);
            return;
        }

        // Weight colors by their probabilities from neighboring collapsed cells
        var weights = CalculateColorWeights(x, y, possible);

        // Weighted random selection
        var selectedColor = WeightedRandomChoice(weights);

        _grid[y, x] = selectedColor;
        _collapsedCells++;

        OnPixelCollapsed?.Invoke(x, y, selectedColor);

        if (delayMs > 0 && !cancellationToken.IsCancellationRequested)
        {
            await Task.Delay(delayMs, cancellationToken);
        }
    }

    private Dictionary<ColorRgb, double> CalculateColorWeights(int x, int y, HashSet<ColorRgb> possibleColors)
    {
        var weights = new Dictionary<ColorRgb, double>();

        // Initialize all possible colors with base weight
        foreach (var color in possibleColors)
        {
            weights[color] = 1.0;
        }

        // Check all 8 neighbors
        var directions = new[]
        {
            (Direction.North, 0, -1),
            (Direction.NorthEast, 1, -1),
            (Direction.East, 1, 0),
            (Direction.SouthEast, 1, 1),
            (Direction.South, 0, 1),
            (Direction.SouthWest, -1, 1),
            (Direction.West, -1, 0),
            (Direction.NorthWest, -1, -1)
        };

        foreach (var (dir, dx, dy) in directions)
        {
            var nx = x + dx;
            var ny = y + dy;

            if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;
            if (!_grid[ny, nx].HasValue) continue; // Not collapsed yet

            var neighborColor = _grid[ny, nx].Value;
            var normalizedX = (float)nx / Math.Max(_width - 1, 1);
            var normalizedY = (float)ny / Math.Max(_height - 1, 1);

            // Get weighted predictions from neighbor
            var predictions = _graph.GetWeightedNeighbors(neighborColor, normalizedX, normalizedY, dir);

            // Boost weights for colors predicted by this neighbor
            foreach (var (color, weight) in predictions)
            {
                if (weights.ContainsKey(color))
                {
                    weights[color] += weight * 10.0; // Strong influence from trained patterns
                }
            }
        }

        return weights;
    }

    private ColorRgb WeightedRandomChoice(Dictionary<ColorRgb, double> weights)
    {
        var totalWeight = weights.Values.Sum();

        if (totalWeight <= 0)
        {
            // Fallback to uniform random
            var colors = weights.Keys.ToArray();
            return colors[_random.Next(colors.Length)];
        }

        var rand = _random.NextDouble() * totalWeight;
        var cumulative = 0.0;

        foreach (var (color, weight) in weights)
        {
            cumulative += weight;
            if (rand <= cumulative)
            {
                return color;
            }
        }

        // Fallback (shouldn't reach here)
        return weights.Keys.First();
    }

    private void PropagateConstraints(int x, int y)
    {
        var collapsedColor = _grid[y, x];
        if (!collapsedColor.HasValue) return;

        var normalizedX = (float)x / Math.Max(_width - 1, 1);
        var normalizedY = (float)y / Math.Max(_height - 1, 1);

        // Propagate to all 8 neighbors
        var directions = new[]
        {
            (Direction.North, 0, -1),
            (Direction.NorthEast, 1, -1),
            (Direction.East, 1, 0),
            (Direction.SouthEast, 1, 1),
            (Direction.South, 0, 1),
            (Direction.SouthWest, -1, 1),
            (Direction.West, -1, 0),
            (Direction.NorthWest, -1, -1)
        };

        foreach (var (dir, dx, dy) in directions)
        {
            var nx = x + dx;
            var ny = y + dy;

            if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;
            if (_grid[ny, nx].HasValue) continue; // Already collapsed

            // Get possible colors from graph
            var predictions = _graph.GetWeightedNeighbors(collapsedColor.Value, normalizedX, normalizedY, dir);

            if (predictions.Count > 0)
            {
                // Constrain neighbor to only colors predicted by graph
                var validColors = new HashSet<ColorRgb>(predictions.Select(p => p.color));
                _possibleColors[ny, nx].IntersectWith(validColors);
            }
        }
    }

    private async Task FillRemaining(int delayMs, CancellationToken cancellationToken)
    {
        var allColors = _graph.GetAllColors();

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_grid[y, x].HasValue) continue;

                var color = allColors[_random.Next(allColors.Count)];
                _grid[y, x] = color;
                _collapsedCells++;

                OnPixelCollapsed?.Invoke(x, y, color);

                if (delayMs > 0 && !cancellationToken.IsCancellationRequested)
                {
                    await Task.Delay(delayMs, cancellationToken);
                }
            }
        }
    }
}
