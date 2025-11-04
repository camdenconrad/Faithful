using System;
using System.Collections.Generic;
using System.Linq;

namespace NNImage.Models;

/// <summary>
/// Represents a neighborhood pattern around a pixel
/// Used for context-aware pattern matching
/// </summary>
public readonly struct NeighborhoodPattern : IEquatable<NeighborhoodPattern>
{
    public ColorRgb Center { get; }
    public Dictionary<Direction, ColorRgb?> Neighbors { get; }

    public NeighborhoodPattern(ColorRgb center, Dictionary<Direction, ColorRgb?> neighbors)
    {
        Center = center;
        Neighbors = neighbors;
    }

    public bool Equals(NeighborhoodPattern other)
    {
        if (!Center.Equals(other.Center))
            return false;

        if (Neighbors.Count != other.Neighbors.Count)
            return false;

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            var thisColor = Neighbors.GetValueOrDefault(dir);
            var otherColor = other.Neighbors.GetValueOrDefault(dir);

            if (!Equals(thisColor, otherColor))
                return false;
        }

        return true;
    }

    public override bool Equals(object? obj)
    {
        return obj is NeighborhoodPattern other && Equals(other);
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Center);

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            hash.Add(Neighbors.GetValueOrDefault(dir));
        }

        return hash.ToHashCode();
    }

    /// <summary>
    /// Calculate similarity score between this pattern and another
    /// Returns 0.0-1.0 where 1.0 is identical
    /// </summary>
    public double CalculateSimilarity(NeighborhoodPattern other)
    {
        if (!Center.Equals(other.Center))
            return 0.0;

        int matches = 0;
        int total = 0;

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            var thisColor = Neighbors.GetValueOrDefault(dir);
            var otherColor = other.Neighbors.GetValueOrDefault(dir);

            if (thisColor != null && otherColor != null)
            {
                total++;
                if (thisColor.Value.Equals(otherColor.Value))
                    matches++;
            }
        }

        return total > 0 ? (double)matches / total : 0.0;
    }

    public override string ToString()
    {
        return $"Pattern[Center={Center}, Neighbors={Neighbors.Count}]";
    }
}
