using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NNImage.Models;

/// <summary>
/// Ultra-fast graph for SUPER FAST training
/// Simple node-based structure with weighted edges
/// No complex nested dictionaries - just nodes and edges!
/// </summary>
public class FastContextGraph
{
    // All nodes in the graph
    private readonly List<GraphNode> _allNodes = new();

    // Fast lookup: color -> list of nodes with that color
    private readonly Dictionary<ColorRgb, List<GraphNode>> _nodesByColor = new();

    // Spatial hash grid for ultra-fast position-based lookup
    // Grid cell -> list of nodes in that cell
    private readonly Dictionary<int, List<GraphNode>> _spatialGrid = new();
    private const int GRID_SIZE = 16; // 16x16 spatial grid

    // Simple statistics
    private int _totalEdges = 0;

    public FastContextGraph()
    {
        Console.WriteLine("[FastContextGraph] Initialized ultra-fast node-based graph");
    }

    /// <summary>
    /// Add or get a node at the specified color and position
    /// SUPER FAST - uses spatial hashing
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GraphNode GetOrCreateNode(ColorRgb color, float normalizedX, float normalizedY)
    {
        // Check spatial grid for nearby existing node
        var gridKey = GetGridKey(normalizedX, normalizedY);

        if (_spatialGrid.TryGetValue(gridKey, out var cellNodes))
        {
            // Look for matching node in this grid cell
            for (int i = 0; i < cellNodes.Count; i++)
            {
                var node = cellNodes[i];
                if (node.Color.Equals(color))
                {
                    var dx = node.NormalizedX - normalizedX;
                    var dy = node.NormalizedY - normalizedY;
                    var distSq = dx * dx + dy * dy;

                    // If close enough (within 5% of image), reuse node
                    if (distSq < 0.0025f) // sqrt(0.0025) = 0.05 = 5%
                    {
                        node.ObservationCount++;
                        return node;
                    }
                }
            }
        }

        // Create new node
        var newNode = new GraphNode(color, normalizedX, normalizedY);
        _allNodes.Add(newNode);

        // Add to color index
        if (!_nodesByColor.ContainsKey(color))
        {
            _nodesByColor[color] = new List<GraphNode>();
        }
        _nodesByColor[color].Add(newNode);

        // Add to spatial grid
        if (!_spatialGrid.ContainsKey(gridKey))
        {
            _spatialGrid[gridKey] = new List<GraphNode>();
        }
        _spatialGrid[gridKey].Add(newNode);

        return newNode;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetGridKey(float normalizedX, float normalizedY)
    {
        var gridX = (int)(normalizedX * GRID_SIZE);
        var gridY = (int)(normalizedY * GRID_SIZE);
        gridX = Math.Clamp(gridX, 0, GRID_SIZE - 1);
        gridY = Math.Clamp(gridY, 0, GRID_SIZE - 1);
        return gridY * GRID_SIZE + gridX;
    }

    /// <summary>
    /// Add edge between two nodes - SUPER FAST
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddEdge(GraphNode fromNode, Direction direction, GraphNode toNode, float weight = 1.0f)
    {
        fromNode.AddEdge(direction, toNode, weight);
        _totalEdges++;
    }

    /// <summary>
    /// Find nodes matching a color - returns all nodes with that color
    /// </summary>
    public List<GraphNode> FindNodesByColor(ColorRgb color)
    {
        if (_nodesByColor.TryGetValue(color, out var nodes))
        {
            return nodes;
        }
        return new List<GraphNode>();
    }

    /// <summary>
    /// Find nodes near a spatial position
    /// </summary>
    public List<GraphNode> FindNodesNearPosition(float normalizedX, float normalizedY, float radius = 0.1f)
    {
        var results = new List<GraphNode>();
        var radiusSq = radius * radius;

        // Check current grid cell and neighbors
        var centerGridKey = GetGridKey(normalizedX, normalizedY);
        var gridX = centerGridKey % GRID_SIZE;
        var gridY = centerGridKey / GRID_SIZE;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                var checkX = gridX + dx;
                var checkY = gridY + dy;
                if (checkX >= 0 && checkX < GRID_SIZE && checkY >= 0 && checkY < GRID_SIZE)
                {
                    var checkKey = checkY * GRID_SIZE + checkX;
                    if (_spatialGrid.TryGetValue(checkKey, out var cellNodes))
                    {
                        for (int i = 0; i < cellNodes.Count; i++)
                        {
                            var node = cellNodes[i];
                            var dxNode = node.NormalizedX - normalizedX;
                            var dyNode = node.NormalizedY - normalizedY;
                            var distSq = dxNode * dxNode + dyNode * dyNode;

                            if (distSq <= radiusSq)
                            {
                                results.Add(node);
                            }
                        }
                    }
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Get weighted predictions from a node
    /// </summary>
    public List<(ColorRgb color, double weight)> GetWeightedNeighbors(ColorRgb centerColor, float normalizedX, float normalizedY, Direction direction)
    {
        // Find closest matching node
        var candidates = FindNodesByColor(centerColor);
        if (candidates.Count == 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Find closest node by position
        GraphNode? closestNode = null;
        float minDist = float.MaxValue;

        for (int i = 0; i < candidates.Count; i++)
        {
            var node = candidates[i];
            var dx = node.NormalizedX - normalizedX;
            var dy = node.NormalizedY - normalizedY;
            var distSq = dx * dx + dy * dy;

            if (distSq < minDist)
            {
                minDist = distSq;
                closestNode = node;
            }
        }

        if (closestNode == null)
        {
            return new List<(ColorRgb, double)>();
        }

        return closestNode.GetWeightedNeighbors(direction);
    }

    public int GetNodeCount() => _allNodes.Count;
    public int GetEdgeCount() => _totalEdges;
    public int GetColorCount() => _nodesByColor.Count;

    public List<ColorRgb> GetAllColors()
    {
        return _nodesByColor.Keys.ToList();
    }

    public void PrintStats()
    {
        Console.WriteLine($"[FastContextGraph] Nodes: {_allNodes.Count:N0}, Edges: {_totalEdges:N0}, Colors: {_nodesByColor.Count}");
        Console.WriteLine($"[FastContextGraph] Avg edges per node: {(_allNodes.Count > 0 ? _totalEdges / (double)_allNodes.Count : 0):F1}");
        Console.WriteLine($"[FastContextGraph] Spatial grid cells used: {_spatialGrid.Count}/{GRID_SIZE * GRID_SIZE}");
    }
}
