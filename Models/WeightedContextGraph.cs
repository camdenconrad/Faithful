using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;

namespace NNImage.Models;

/// <summary>
/// Context-aware weighted graph that considers neighborhood patterns
/// Much more powerful than simple adjacency
/// </summary>
public class WeightedContextGraph
{
    // pattern -> direction -> target_color -> weight
    private readonly ConcurrentDictionary<NeighborhoodPattern, Dictionary<Direction, Dictionary<ColorRgb, double>>> _contextPatterns = new();

    // Fallback: Simple adjacency for when we don't have enough context
    private readonly ConcurrentDictionary<ColorRgb, ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>> _simpleGraph = new();

    private readonly object _normalizeLock = new object();
    private bool _isNormalized;

    public void AddPattern(NeighborhoodPattern pattern, Direction outputDirection, ColorRgb targetColor, double weight = 1.0)
    {
        var patternDict = _contextPatterns.GetOrAdd(pattern, _ => new Dictionary<Direction, Dictionary<ColorRgb, double>>());

        lock (patternDict)
        {
            if (!patternDict.ContainsKey(outputDirection))
                patternDict[outputDirection] = new Dictionary<ColorRgb, double>();

            if (!patternDict[outputDirection].ContainsKey(targetColor))
                patternDict[outputDirection][targetColor] = 0;

            patternDict[outputDirection][targetColor] += weight;
        }

        _isNormalized = false;
    }

    public void AddSimpleAdjacency(ColorRgb centerColor, Direction direction, ColorRgb neighborColor)
    {
        var colorDict = _simpleGraph.GetOrAdd(centerColor, _ => new ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>());
        var directionDict = colorDict.GetOrAdd(direction, _ => new ConcurrentDictionary<ColorRgb, int>());
        directionDict.AddOrUpdate(neighborColor, 1, (_, count) => count + 1);

        _isNormalized = false;
    }

    public void Normalize()
    {
        lock (_normalizeLock)
        {
            if (_isNormalized)
                return;

            Console.WriteLine("[WeightedContextGraph] Normalizing weights...");

            // Normalize context patterns
            foreach (var pattern in _contextPatterns.Keys.ToList())
            {
                var directions = _contextPatterns[pattern];
                foreach (var dir in directions.Keys.ToList())
                {
                    var colors = directions[dir];
                    var total = colors.Values.Sum();

                    if (total > 0)
                    {
                        foreach (var color in colors.Keys.ToList())
                        {
                            colors[color] /= total;
                        }
                    }
                }
            }

            _isNormalized = true;
            Console.WriteLine($"[WeightedContextGraph] Normalized {_contextPatterns.Count} context patterns");
        }
    }

    public List<(ColorRgb color, double weight)> GetWeightedNeighbors(NeighborhoodPattern currentPattern, Direction direction)
    {
        if (!_isNormalized)
            Normalize();

        var results = new List<(ColorRgb color, double weight)>();

        // Try exact pattern match first
        if (_contextPatterns.TryGetValue(currentPattern, out var patternDirs) &&
            patternDirs.TryGetValue(direction, out var colors))
        {
            results.AddRange(colors.Select(kvp => (kvp.Key, kvp.Value)));
            return results;
        }

        // Try similar patterns with weighted blending
        var similarPatterns = _contextPatterns.Keys
            .Select(p => (pattern: p, similarity: currentPattern.CalculateSimilarity(p)))
            .Where(x => x.similarity > 0.5) // Only use reasonably similar patterns
            .OrderByDescending(x => x.similarity)
            .Take(5) // Top 5 most similar
            .ToList();

        if (similarPatterns.Any())
        {
            var weightedColors = new Dictionary<ColorRgb, double>();

            foreach (var (pattern, similarity) in similarPatterns)
            {
                if (_contextPatterns[pattern].TryGetValue(direction, out var patternColors))
                {
                    foreach (var (color, weight) in patternColors)
                    {
                        if (!weightedColors.ContainsKey(color))
                            weightedColors[color] = 0;

                        weightedColors[color] += weight * similarity;
                    }
                }
            }

            // Normalize combined weights
            var total = weightedColors.Values.Sum();
            if (total > 0)
            {
                results.AddRange(weightedColors.Select(kvp => (kvp.Key, kvp.Value / total)));
            }
        }

        // Fallback to simple adjacency if no context available
        if (results.Count == 0 && _simpleGraph.TryGetValue(currentPattern.Center, out var centerDict) &&
            centerDict.TryGetValue(direction, out var neighbors))
        {
            var total = neighbors.Values.Sum();
            results.AddRange(neighbors.Select(kvp => (kvp.Key, (double)kvp.Value / total)));
        }

        return results;
    }

    public List<ColorRgb> GetPossibleNeighbors(NeighborhoodPattern pattern, Direction direction)
    {
        var weighted = GetWeightedNeighbors(pattern, direction);
        return weighted.Select(x => x.color).ToList();
    }

    public List<ColorRgb> GetAllColors()
    {
        var colors = new HashSet<ColorRgb>();

        foreach (var pattern in _contextPatterns.Keys)
        {
            colors.Add(pattern.Center);
            foreach (var neighbor in pattern.Neighbors.Values)
            {
                if (neighbor.HasValue)
                    colors.Add(neighbor.Value);
            }
        }

        foreach (var color in _simpleGraph.Keys)
        {
            colors.Add(color);
        }

        return colors.ToList();
    }

    public int GetColorCount()
    {
        return GetAllColors().Count;
    }

    public int GetPatternCount()
    {
        return _contextPatterns.Count;
    }
}
