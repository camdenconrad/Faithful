using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNImage.Models;

public class TrainingDataCache
{
    [JsonPropertyName("quantizationLevel")]
    public int QuantizationLevel { get; set; }

    [JsonPropertyName("colorCount")]
    public int ColorCount { get; set; }

    [JsonPropertyName("adjacencies")]
    public Dictionary<string, Dictionary<string, Dictionary<string, int>>> Adjacencies { get; set; } = new();

    [JsonPropertyName("trainingImagePaths")]
    public List<string> TrainingImagePaths { get; set; } = new();

    [JsonPropertyName("lastTrainingDate")]
    public DateTime LastTrainingDate { get; set; }

    private static readonly string CacheFilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "NNImage",
        "training_cache.json");

    public static TrainingDataCache? Load()
    {
        try
        {
            if (!File.Exists(CacheFilePath))
                return null;

            var json = File.ReadAllText(CacheFilePath);
            return JsonSerializer.Deserialize<TrainingDataCache>(json);
        }
        catch
        {
            return null;
        }
    }

    public void Save()
    {
        try
        {
            var directory = Path.GetDirectoryName(CacheFilePath);
            if (directory != null && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(CacheFilePath, json);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to save cache: {ex.Message}");
        }
    }

    public static void Clear()
    {
        try
        {
            if (File.Exists(CacheFilePath))
                File.Delete(CacheFilePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to clear cache: {ex.Message}");
        }
    }

    public AdjacencyGraph ToAdjacencyGraph()
    {
        var graph = new AdjacencyGraph();

        foreach (var (colorKey, directions) in Adjacencies)
        {
            var centerColor = ParseColorKey(colorKey);

            foreach (var (directionKey, neighbors) in directions)
            {
                var direction = Enum.Parse<Direction>(directionKey);

                foreach (var (neighborKey, count) in neighbors)
                {
                    var neighborColor = ParseColorKey(neighborKey);

                    for (int i = 0; i < count; i++)
                    {
                        graph.AddAdjacency(centerColor, direction, neighborColor);
                    }
                }
            }
        }

        graph.Normalize();
        return graph;
    }

    public static TrainingDataCache FromAdjacencyGraph(AdjacencyGraph graph, int quantizationLevel, List<string> imagePaths)
    {
        var cache = new TrainingDataCache
        {
            QuantizationLevel = quantizationLevel,
            ColorCount = graph.GetColorCount(),
            TrainingImagePaths = imagePaths,
            LastTrainingDate = DateTime.Now
        };

        // Export graph data (we need to add a method to AdjacencyGraph to export its data)
        // For now, we'll store it in a serializable format
        var colors = graph.GetAllColors();
        foreach (var color in colors)
        {
            var colorKey = ColorToKey(color);
            cache.Adjacencies[colorKey] = new Dictionary<string, Dictionary<string, int>>();

            foreach (var direction in DirectionExtensions.AllDirections)
            {
                var neighbors = graph.GetPossibleNeighbors(color, direction);
                if (neighbors.Count > 0)
                {
                    var directionKey = direction.ToString();
                    cache.Adjacencies[colorKey][directionKey] = new Dictionary<string, int>();

                    foreach (var neighbor in neighbors)
                    {
                        var neighborKey = ColorToKey(neighbor);
                        cache.Adjacencies[colorKey][directionKey][neighborKey] = 1;
                    }
                }
            }
        }

        return cache;
    }

    private static string ColorToKey(ColorRgb color)
    {
        return $"{color.R},{color.G},{color.B}";
    }

    private static ColorRgb ParseColorKey(string key)
    {
        var parts = key.Split(',');
        return new ColorRgb(
            byte.Parse(parts[0]),
            byte.Parse(parts[1]),
            byte.Parse(parts[2]));
    }
}
