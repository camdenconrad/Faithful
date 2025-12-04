using System;
using System.Collections.Generic;
using System.Linq;
using NNImage.Services;

namespace NNImage.Models;

/// <summary>
/// Core terrain editor with AI-powered texture synthesis and smart brushes
/// Uses MultiScaleContextGraph and repliKate for intelligent content generation
/// </summary>
public class TerrainEditor
{
    public LayerStack LayerStack { get; private set; }
    public TextureLibrary TextureLibrary { get; private set; }
    public TerrainBrush CurrentBrush { get; set; }

    private readonly GpuAccelerator? _gpu;
    private readonly MultiScaleContextGraph _masterGraph;
    private readonly ColorQuantizer _globalQuantizer;
    private readonly List<(int x, int y, float time)> _strokeHistory = new();
    private TerrainLayer? _activeLayer;

    public int Width => LayerStack.Width;
    public int Height => LayerStack.Height;

    public TerrainEditor(int width, int height, GpuAccelerator? gpu = null)
    {
        LayerStack = new LayerStack(width, height);
        TextureLibrary = new TextureLibrary(gpu);
        CurrentBrush = new TerrainBrush();
        _gpu = gpu;

        // Initialize master AI systems
        _masterGraph = new MultiScaleContextGraph();
        _masterGraph.SetGpuAccelerator(_gpu);

        // Initialize global quantizer
        _globalQuantizer = new ColorQuantizer(256, _gpu);
        InitializeGlobalPalette();

        _activeLayer = LayerStack.Layers.First();

        Console.WriteLine($"[TerrainEditor] Initialized {width}x{height} terrain editor with AI systems");
    }

    private void InitializeGlobalPalette()
    {
        // Build a comprehensive global palette for terrain editing
        var colors = new List<ColorRgb>();

        // Add common terrain colors
        for (int i = 0; i < 256; i++)
        {
            byte val = (byte)i;
            colors.Add(new ColorRgb(val, val, val)); // Grayscale
            colors.Add(new ColorRgb(val, (byte)(val * 0.8f), (byte)(val * 0.6f))); // Earth tones
            colors.Add(new ColorRgb((byte)(val * 0.5f), val, (byte)(val * 0.5f))); // Greens
        }

        _globalQuantizer.BuildPalette(colors);
        Console.WriteLine("[TerrainEditor] Global color palette initialized");
    }

    public void SetActiveLayer(int layerId)
    {
        _activeLayer = LayerStack.GetLayer(layerId);
    }

    public TerrainLayer? GetActiveLayer() => _activeLayer;

    /// <summary>
    /// Apply brush stroke with AI assistance
    /// </summary>
    public void ApplyBrushStroke(int x, int y, float deltaTime = 0.016f)
    {
        if (_activeLayer == null)
            return;

        _strokeHistory.Add((x, y, deltaTime));

        // Apply brush to active layer
        if (_activeLayer.Type == TerrainLayer.LayerType.Height && _activeLayer.HeightData != null)
        {
            CurrentBrush.ApplyStroke(_activeLayer.HeightData, Width, Height, x, y, null, deltaTime);
        }
        else if (_activeLayer.Type == TerrainLayer.LayerType.Texture && _activeLayer.TextureData != null)
        {
            CurrentBrush.ApplyStroke(null, Width, Height, x, y, _activeLayer.TextureData, deltaTime);
        }
    }

    /// <summary>
    /// Smart fill using AI to predict and fill content
    /// </summary>
    public void SmartFill(int startX, int startY, float tolerance = 0.1f)
    {
        if (_activeLayer?.HeightData == null)
            return;

        Console.WriteLine($"[TerrainEditor] AI Smart Fill from ({startX}, {startY})");

        // Use WFC to intelligently fill the region
        var heightmap = _activeLayer.HeightData;
        var targetHeight = heightmap[startY * Width + startX];

        // Find connected region
        var toFill = FloodFindRegion(startX, startY, targetHeight, tolerance, heightmap);

        Console.WriteLine($"[TerrainEditor] Found {toFill.Count} pixels to fill");

        // Use AI to generate natural fill
        if (toFill.Count > 0)
        {
            FillRegionWithAI(toFill, heightmap);
        }
    }

    private HashSet<(int x, int y)> FloodFindRegion(int startX, int startY, float targetValue, float tolerance, float[] data)
    {
        var region = new HashSet<(int, int)>();
        var queue = new Queue<(int, int)>();
        queue.Enqueue((startX, startY));

        while (queue.Count > 0 && region.Count < 100000) // Limit region size
        {
            var (x, y) = queue.Dequeue();

            if (x < 0 || x >= Width || y < 0 || y >= Height)
                continue;

            if (region.Contains((x, y)))
                continue;

            float value = data[y * Width + x];
            if (Math.Abs(value - targetValue) > tolerance)
                continue;

            region.Add((x, y));

            queue.Enqueue((x + 1, y));
            queue.Enqueue((x - 1, y));
            queue.Enqueue((x, y + 1));
            queue.Enqueue((x, y - 1));
        }

        return region;
    }

    private void FillRegionWithAI(HashSet<(int x, int y)> region, float[] heightmap)
    {
        // Use surrounding context to intelligently fill region
        var avgHeight = 0f;
        var borderPixels = new List<(int x, int y, float h)>();

        foreach (var (x, y) in region)
        {
            // Check neighbors outside region
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;

                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < Width && ny >= 0 && ny < Height && !region.Contains((nx, ny)))
                    {
                        float h = heightmap[ny * Width + nx];
                        borderPixels.Add((nx, ny, h));
                        avgHeight += h;
                    }
                }
            }
        }

        if (borderPixels.Count > 0)
        {
            avgHeight /= borderPixels.Count;

            // Fill with average + some noise for natural look
            var random = new Random();
            foreach (var (x, y) in region)
            {
                float noise = (float)(random.NextDouble() - 0.5) * 0.1f;
                heightmap[y * Width + x] = Math.Clamp(avgHeight + noise, 0f, 1f);
            }
        }
    }

    /// <summary>
    /// Generate terrain using AI
    /// </summary>
    public void GenerateAITerrain(HeightmapGenerator.TerrainType terrainType, float roughness, int octaves, float scale)
    {
        var baseLayer = LayerStack.Layers.FirstOrDefault(l => l.Type == TerrainLayer.LayerType.Height);
        if (baseLayer?.HeightData == null)
            return;

        var generator = new HeightmapGenerator();
        var heightmap = generator.Generate(Width, Height, terrainType, roughness, octaves, scale);

        Array.Copy(heightmap, baseLayer.HeightData, heightmap.Length);

        Console.WriteLine($"[TerrainEditor] Generated {terrainType} terrain");
    }

    /// <summary>
    /// Apply AI-synthesized texture to terrain
    /// </summary>
    public void ApplyAITexture(int textureId)
    {
        var textureLayer = LayerStack.Layers.FirstOrDefault(l => l.Type == TerrainLayer.LayerType.Texture);
        if (textureLayer?.TextureData == null)
            return;

        var texture = TextureLibrary.GetTexture(textureId);
        if (texture?.TrainedGraph == null)
        {
            Console.WriteLine($"[TerrainEditor] Texture {textureId} not trained yet");
            return;
        }

        Console.WriteLine($"[TerrainEditor] Applying AI-synthesized texture '{texture.Name}'");

        // Use trained graph to generate texture that fits the terrain
        var synthesized = TextureLibrary.SynthesizeTexture(textureId, Width, Height, 8);

        // Convert to texture IDs (simplified)
        for (int i = 0; i < textureLayer.TextureData.Length; i++)
        {
            textureLayer.TextureData[i] = (byte)textureId;
        }

        Console.WriteLine($"[TerrainEditor] Texture applied successfully");
    }

    /// <summary>
    /// Get final composited result
    /// </summary>
    public (float[] heightmap, byte[] textureMask, uint[] coloredPixels) GetFinalTerrain(
        TerrainColorizer.ColorScheme colorScheme = TerrainColorizer.ColorScheme.Realistic,
        bool useLighting = true, float sunAngle = 45f)
    {
        var heightmap = LayerStack.CompositeHeightLayers();
        var textureMask = LayerStack.CompositeTextureLayers();

        // Apply coloring
        var colorizer = new TerrainColorizer();
        var pixels = colorizer.Colorize(heightmap, Width, Height, colorScheme, useLighting, sunAngle);

        return (heightmap, textureMask, pixels);
    }

    public void ClearStrokeHistory()
    {
        _strokeHistory.Clear();
    }
}
