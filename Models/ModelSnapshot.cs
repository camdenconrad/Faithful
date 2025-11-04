using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using NNImage.Services;

namespace NNImage.Models;

// DTOs for persisting the trained model (single-scale 9x9 at the moment)
public class ModelSnapshot
{
    [JsonPropertyName("version")] public int Version { get; set; } = 1;
    [JsonPropertyName("scales")] public int[] Scales { get; set; } = new[] { 8 }; // radii (single-scale radius 8 = 17x17 for larger context)
    [JsonPropertyName("graphs")] public Dictionary<int, WeightedContextGraphSnapshot> Graphs { get; set; } = new();
    [JsonPropertyName("created")] public DateTime Created { get; set; } = DateTime.Now;
    [JsonPropertyName("notes")] public string? Notes { get; set; }
}

public class WeightedContextGraphSnapshot
{
    // Normalized pattern distributions: each entry is one NeighborhoodPattern with per-direction color weights
    [JsonPropertyName("patterns")] public List<PatternEntry> Patterns { get; set; } = new();

    // Simple adjacency normalized distributions (fallback)
    [JsonPropertyName("adjacency")] public List<SimpleAdjacencyEntry> Adjacency { get; set; } = new();

    // Known distinct colors (optional; speeds up GetAllColors)
    [JsonPropertyName("colors")] public List<ColorRgb> Colors { get; set; } = new();
}

public class PatternEntry
{
    [JsonPropertyName("center")] public ColorRgb Center { get; set; }
    // neighbors length 8, null means out-of-bounds
    [JsonPropertyName("neighbors")] public ColorRgb?[] Neighbors { get; set; } = new ColorRgb?[8];
    // Direction -> color weights
    [JsonPropertyName("dir")] public List<DirectionWeightsEntry> DirectionWeights { get; set; } = new();
}

public class DirectionWeightsEntry
{
    [JsonPropertyName("direction")] public Direction Direction { get; set; }
    [JsonPropertyName("colors")] public List<ColorWeight> Colors { get; set; } = new();
}

public class ColorWeight
{
    [JsonPropertyName("color")] public ColorRgb Color { get; set; }
    [JsonPropertyName("w")] public double Weight { get; set; }
}

public class SimpleAdjacencyEntry
{
    [JsonPropertyName("center")] public ColorRgb Center { get; set; }
    [JsonPropertyName("direction")] public Direction Direction { get; set; }
    [JsonPropertyName("colors")] public List<ColorWeight> Colors { get; set; } = new();
}

public static class ModelRepository
{
    private static string DefaultPath
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("NNIMAGE_MODEL_PATH");
            if (!string.IsNullOrWhiteSpace(env)) return env!;
            var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "NNImage");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "model_v1.json");
        }
    }

    public static bool Save(MultiScaleContextGraph graph, string? path = null, string? notes = null)
    {
        try
        {
            var snapshot = graph.ToSnapshot();
            snapshot.Notes = notes;
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                IncludeFields = false
            };
            string json = System.Text.Json.JsonSerializer.Serialize<ModelSnapshot>(snapshot, options);
            var file = path ?? DefaultPath;
            var dir = Path.GetDirectoryName(file);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(file, json);
            Console.WriteLine($"[ModelRepository] Saved model to {file} (patterns: {graph.GetTotalPatternCount()})");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Save failed: {ex.Message}");
            return false;
        }
    }

    public static MultiScaleContextGraph? Load(string? path = null, GpuAccelerator? gpu = null)
    {
        try
        {
            var file = path ?? DefaultPath;
            if (!File.Exists(file))
            {
                Console.WriteLine($"[ModelRepository] No model file at {file}");
                return null;
            }
            var json = File.ReadAllText(file);
            var options = new JsonSerializerOptions();
            var snapshot = JsonSerializer.Deserialize<ModelSnapshot>(json, options);
            if (snapshot == null)
            {
                Console.WriteLine("[ModelRepository] Snapshot deserialized to null");
                return null;
            }
            if (snapshot.Version != 1)
            {
                Console.WriteLine($"[ModelRepository] Unsupported snapshot version: {snapshot.Version}");
                return null;
            }
            // Support single-scale radius 8 (17x17) for large context learning
            if (snapshot.Scales == null || snapshot.Scales.Length == 0 || snapshot.Scales.Any(s => s != 8 && s != 4))
            {
                Console.WriteLine("[ModelRepository] Snapshot scales not strictly matching current single-scale (8); attempting best-effort load");
            }

            var graph = MultiScaleContextGraph.FromSnapshot(snapshot, gpu);
            return graph;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Load failed: {ex.Message}");
            return null;
        }
    }
}
