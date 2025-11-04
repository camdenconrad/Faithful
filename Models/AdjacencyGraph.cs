using System;
using System.Collections.Generic;
using System.Linq;

namespace NNImage;

public class AdjacencyGraph
{
    // adjacency_graph[color][direction][neighbor_color] = count
    private readonly Dictionary<ColorRgb, Dictionary<Direction, Dictionary<ColorRgb, int>>> _graph = new();
    private readonly Dictionary<ColorRgb, Dictionary<Direction, Dictionary<ColorRgb, double>>> _probabilities = new();
    private bool _isNormalized;

    public void AddAdjacency(ColorRgb centerColor, Direction direction, ColorRgb neighborColor)
    {
        if (!_graph.ContainsKey(centerColor))
            _graph[centerColor] = new Dictionary<Direction, Dictionary<ColorRgb, int>>();

        if (!_graph[centerColor].ContainsKey(direction))
            _graph[centerColor][direction] = new Dictionary<ColorRgb, int>();

        if (!_graph[centerColor][direction].ContainsKey(neighborColor))
            _graph[centerColor][direction][neighborColor] = 0;

        _graph[centerColor][direction][neighborColor]++;
        _isNormalized = false;
    }

    public void Normalize()
    {
        _probabilities.Clear();

        foreach (var (color, directions) in _graph)
        {
            _probabilities[color] = new Dictionary<Direction, Dictionary<ColorRgb, double>>();

            foreach (var (direction, neighbors) in directions)
            {
                var total = neighbors.Values.Sum();
                _probabilities[color][direction] = new Dictionary<ColorRgb, double>();

                foreach (var (neighborColor, count) in neighbors)
                {
                    _probabilities[color][direction][neighborColor] = (double)count / total;
                }
            }
        }

        _isNormalized = true;
    }

    public ColorRgb? GetRandomNeighbor(ColorRgb centerColor, Direction direction, Random random)
    {
        if (!_isNormalized)
            Normalize();

        if (!_probabilities.ContainsKey(centerColor) || 
            !_probabilities[centerColor].ContainsKey(direction))
            return null;

        var neighbors = _probabilities[centerColor][direction];
        if (neighbors.Count == 0)
            return null;

        var rand = random.NextDouble();
        var cumulative = 0.0;

        foreach (var (neighborColor, probability) in neighbors)
        {
            cumulative += probability;
            if (rand <= cumulative)
                return neighborColor;
        }

        // Fallback to last color
        return neighbors.Keys.Last();
    }

    public List<ColorRgb> GetPossibleNeighbors(ColorRgb centerColor, Direction direction)
    {
        if (!_graph.ContainsKey(centerColor) || 
            !_graph[centerColor].ContainsKey(direction))
            return new List<ColorRgb>();

        return _graph[centerColor][direction].Keys.ToList();
    }

    public List<ColorRgb> GetAllColors()
    {
        return _graph.Keys.ToList();
    }

    public int GetColorCount()
    {
        return _graph.Keys.Count;
    }

    public bool HasColor(ColorRgb color)
    {
        return _graph.ContainsKey(color);
    }
}
