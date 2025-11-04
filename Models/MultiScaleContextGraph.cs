using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using NNImage.Services;

namespace NNImage.Models;

/// <summary>
/// Multi-scale context graph that learns patterns at different scales
/// Combines 3x3, 5x5, and 9x9 neighborhoods for hierarchical understanding
/// </summary>
public class MultiScaleContextGraph
{
    private readonly Dictionary<int, WeightedContextGraph> _scaleGraphs = new();
    private readonly Dictionary<ColorRgb, (float dx, float dy)> _directionalBias = new();
    private readonly int[] _scales = { 1, 2, 4 }; // Radii: 3x3, 5x5, 9x9
    private readonly Dictionary<ColorRgb, ColorRgb> _featureMap = new(); // Color -> Representative color
    private GpuAccelerator? _gpu;

    public MultiScaleContextGraph()
    {
        foreach (var scale in _scales)
        {
            _scaleGraphs[scale] = new WeightedContextGraph();
        }
        Console.WriteLine($"[MultiScale] Initialized with scales: {string.Join(", ", _scales.Select(s => $"{s*2+1}x{s*2+1}"))}");
    }

    public void SetGpuAccelerator(GpuAccelerator? gpu)
    {
        _gpu = gpu;
        foreach (var graph in _scaleGraphs.Values)
        {
            graph.SetGpuAccelerator(gpu);
        }
    }

    public void AddPatternMultiScale(
        ColorRgb centerColor,
        Dictionary<Direction, ColorRgb?> neighbors,
        Direction outputDirection,
        ColorRgb targetColor,
        uint[] pixelRegion,
        int regionWidth,
        int regionHeight,
        int centerX,
        int centerY)
    {
        // 1. Add to standard scale (3x3)
        var pattern3x3 = new NeighborhoodPattern(centerColor, neighbors);
        _scaleGraphs[1].AddPattern(pattern3x3, outputDirection, targetColor);
        _scaleGraphs[1].AddSimpleAdjacency(centerColor, outputDirection, targetColor);

        // 2. Extract and add 5x5 pattern
        if (regionWidth >= 5 && regionHeight >= 5 && 
            centerX >= 2 && centerX < regionWidth - 2 &&
            centerY >= 2 && centerY < regionHeight - 2)
        {
            var neighbors5x5 = ExtractNeighborhood(pixelRegion, regionWidth, regionHeight, centerX, centerY, 2);
            var pattern5x5 = new NeighborhoodPattern(centerColor, neighbors5x5);
            _scaleGraphs[2].AddPattern(pattern5x5, outputDirection, targetColor, 0.8); // Slightly lower weight
        }

        // 3. Extract and add 9x9 pattern
        if (regionWidth >= 9 && regionHeight >= 9 &&
            centerX >= 4 && centerX < regionWidth - 4 &&
            centerY >= 4 && centerY < regionHeight - 4)
        {
            var neighbors9x9 = ExtractNeighborhood(pixelRegion, regionWidth, regionHeight, centerX, centerY, 4);
            var pattern9x9 = new NeighborhoodPattern(centerColor, neighbors9x9);
            _scaleGraphs[4].AddPattern(pattern9x9, outputDirection, targetColor, 0.6); // Even lower weight
        }

        // 4. Calculate and store directional bias (gradient)
        UpdateDirectionalBias(centerColor, outputDirection, targetColor);
    }

    private Dictionary<Direction, ColorRgb?> ExtractNeighborhood(
        uint[] pixels, int width, int height, int centerX, int centerY, int radius)
    {
        var neighbors = new Dictionary<Direction, ColorRgb?>();

        // Sample at the 8 cardinal/diagonal directions at the given radius
        var offsets = new[]
        {
            (0, -radius),      // North
            (radius, -radius), // NorthEast
            (radius, 0),       // East
            (radius, radius),  // SouthEast
            (0, radius),       // South
            (-radius, radius), // SouthWest
            (-radius, 0),      // West
            (-radius, -radius) // NorthWest
        };

        for (int i = 0; i < 8; i++)
        {
            var (dx, dy) = offsets[i];
            var x = centerX + dx;
            var y = centerY + dy;

            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                var pixel = pixels[y * width + x];
                neighbors[(Direction)i] = PixelToColor(pixel);
            }
            else
            {
                neighbors[(Direction)i] = null;
            }
        }

        return neighbors;
    }

    private void UpdateDirectionalBias(ColorRgb centerColor, Direction direction, ColorRgb targetColor)
    {
        // Calculate gradient vector based on color change
        var (dx, dy) = direction.GetOffset();

        // Color difference as gradient magnitude
        var dr = targetColor.R - centerColor.R;
        var dg = targetColor.G - centerColor.G;
        var db = targetColor.B - centerColor.B;
        var magnitude = (float)Math.Sqrt(dr * dr + dg * dg + db * db) / 441.0f; // Normalize by max distance

        // Update moving average of directional bias
        if (!_directionalBias.ContainsKey(centerColor))
            _directionalBias[centerColor] = (0, 0);

        var current = _directionalBias[centerColor];
        var alpha = 0.1f; // Learning rate for bias
        _directionalBias[centerColor] = (
            current.dx * (1 - alpha) + dx * magnitude * alpha,
            current.dy * (1 - alpha) + dy * magnitude * alpha
        );
    }

    public List<(ColorRgb color, double weight)> GetWeightedNeighborsMultiScale(
        NeighborhoodPattern currentPattern,
        Direction direction)
    {
        // Blend predictions from all scales
        var scaleWeights = new Dictionary<int, double>
        {
            { 1, 0.5 },  // 3x3: High weight for local detail
            { 2, 0.3 },  // 5x5: Medium weight for mid-range structure
            { 4, 0.2 }   // 9x9: Lower weight for global coherence
        };

        var combinedPredictions = new Dictionary<ColorRgb, double>();

        foreach (var (scale, scaleWeight) in scaleWeights)
        {
            var scalePredictions = _scaleGraphs[scale].GetWeightedNeighbors(currentPattern, direction);

            foreach (var (color, weight) in scalePredictions)
            {
                if (!combinedPredictions.ContainsKey(color))
                    combinedPredictions[color] = 0;

                combinedPredictions[color] += weight * scaleWeight;
            }
        }

        // Apply directional bias
        var centerColor = currentPattern.Center;
        if (_directionalBias.TryGetValue(centerColor, out var bias))
        {
            var (dx, dy) = direction.GetOffset();
            var dirAlignment = dx * bias.dx + dy * bias.dy; // Dot product

            if (dirAlignment > 0) // If direction aligns with learned bias
            {
                // Boost probabilities of colors in this direction
                foreach (var color in combinedPredictions.Keys.ToList())
                {
                    combinedPredictions[color] *= (1 + dirAlignment * 0.3); // Up to 30% boost
                }
            }
        }

        // Normalize and return
        var total = combinedPredictions.Values.Sum();
        if (total > 0)
        {
            return combinedPredictions
                .Select(kvp => (kvp.Key, kvp.Value / total))
                .OrderByDescending(x => x.Item2)
                .ToList();
        }

        return new List<(ColorRgb, double)>();
    }

    public void Normalize()
    {
        Console.WriteLine("[MultiScale] Normalizing all scale graphs...");
        foreach (var (scale, graph) in _scaleGraphs)
        {
            graph.Normalize();
            Console.WriteLine($"[MultiScale] Scale {scale*2+1}x{scale*2+1}: {graph.GetPatternCount()} patterns");
        }
    }

    public int GetTotalPatternCount()
    {
        return _scaleGraphs.Values.Sum(g => g.GetPatternCount());
    }

    public int GetColorCount()
    {
        return _scaleGraphs[1].GetColorCount();
    }

    public List<ColorRgb> GetAllColors()
    {
        return _scaleGraphs[1].GetAllColors();
    }

    private ColorRgb PixelToColor(uint pixel)
    {
        var a = (byte)((pixel >> 24) & 0xFF);
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);

        if (a < 255)
        {
            var alpha = a / 255.0;
            r = (byte)(r * alpha + 255 * (1 - alpha));
            g = (byte)(g * alpha + 255 * (1 - alpha));
            b = (byte)(b * alpha + 255 * (1 - alpha));
        }

        return new ColorRgb(r, g, b);
    }

    public Dictionary<int, WeightedContextGraph> GetScaleGraphs() => _scaleGraphs;
}
