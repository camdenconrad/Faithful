using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace NNImage.Models;

/// <summary>
/// Ultra-fast graph node for super fast training
/// Stores color, position, and weighted edges to other nodes
/// </summary>
public class GraphNode
{
    public ColorRgb Color { get; }
    public float NormalizedX { get; }
    public float NormalizedY { get; }

    // Weighted edges to neighbor nodes in each direction
    // Direction -> (target_node, weight)
    public Dictionary<Direction, List<(GraphNode node, float weight)>> Edges { get; }

    // Fast lookup: which nodes connect TO this node
    public List<GraphNode> IncomingNodes { get; }

    // Observation count for this node
    public int ObservationCount { get; set; }

    public GraphNode(ColorRgb color, float normalizedX, float normalizedY)
    {
        Color = color;
        NormalizedX = normalizedX;
        NormalizedY = normalizedY;
        Edges = new Dictionary<Direction, List<(GraphNode, float)>>(8);
        IncomingNodes = new List<GraphNode>();
        ObservationCount = 1;
    }

    /// <summary>
    /// Add or update weighted edge to another node
    /// Super fast - just list operations
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddEdge(Direction direction, GraphNode targetNode, float weight = 1.0f)
    {
        if (!Edges.ContainsKey(direction))
        {
            Edges[direction] = new List<(GraphNode, float)>();
        }

        var edges = Edges[direction];

        // Check if edge already exists
        for (int i = 0; i < edges.Count; i++)
        {
            if (ReferenceEquals(edges[i].node, targetNode))
            {
                // Update existing edge weight
                edges[i] = (targetNode, edges[i].weight + weight);
                return;
            }
        }

        // Add new edge
        edges.Add((targetNode, weight));
        targetNode.IncomingNodes.Add(this);
    }

    /// <summary>
    /// Get weighted neighbor predictions for a direction
    /// Returns normalized probability distribution
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public List<(ColorRgb color, double weight)> GetWeightedNeighbors(Direction direction)
    {
        if (!Edges.TryGetValue(direction, out var edges) || edges.Count == 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Calculate total weight
        float totalWeight = 0;
        for (int i = 0; i < edges.Count; i++)
        {
            totalWeight += edges[i].weight;
        }

        if (totalWeight <= 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Normalize and return
        var result = new List<(ColorRgb, double)>(edges.Count);
        for (int i = 0; i < edges.Count; i++)
        {
            var (node, weight) = edges[i];
            result.Add((node.Color, weight / totalWeight));
        }

        return result;
    }

    /// <summary>
    /// Calculate spatial distance to another node
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float SpatialDistance(GraphNode other)
    {
        var dx = NormalizedX - other.NormalizedX;
        var dy = NormalizedY - other.NormalizedY;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    public override string ToString()
    {
        return $"Node[{Color} @ ({NormalizedX:F2}, {NormalizedY:F2}), Edges: {Edges.Values.Sum(e => e.Count)}]";
    }
}
