using System;
using System.Collections.Generic;
using System.Linq;
using NNImage.Services;

namespace NNImage.Models;

/// <summary>
/// Texture library with AI-powered texture synthesis using MultiScaleContextGraph
/// </summary>
public class TextureLibrary
{
    public class TextureEntry
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public uint[] Pixels { get; set; } = Array.Empty<uint>();
        public int Width { get; set; }
        public int Height { get; set; }
        public MultiScaleContextGraph? TrainedGraph { get; set; }
        public ColorQuantizer? Quantizer { get; set; }
        public string Category { get; set; } = "General";
        public Dictionary<string, float> Properties { get; set; } = new();
    }

    private readonly List<TextureEntry> _textures = new();
    private readonly GpuAccelerator? _gpu;
    private int _nextId = 1;

    public TextureLibrary(GpuAccelerator? gpu = null)
    {
        _gpu = gpu;
        InitializeDefaultTextures();
    }

    public IReadOnlyList<TextureEntry> Textures => _textures;

    private void InitializeDefaultTextures()
    {
        // Add default procedural textures
        AddProceduralTexture("Grass", 256, 256, GenerateGrassTexture);
        AddProceduralTexture("Rock", 256, 256, GenerateRockTexture);
        AddProceduralTexture("Sand", 256, 256, GenerateSandTexture);
        AddProceduralTexture("Snow", 256, 256, GenerateSnowTexture);
        AddProceduralTexture("Dirt", 256, 256, GenerateDirtTexture);
        AddProceduralTexture("Stone", 256, 256, GenerateStoneTexture);
    }

    public int AddTexture(string name, uint[] pixels, int width, int height, string category = "Custom")
    {
        var entry = new TextureEntry
        {
            Id = _nextId++,
            Name = name,
            Pixels = pixels,
            Width = width,
            Height = height,
            Category = category
        };

        _textures.Add(entry);
        return entry.Id;
    }

    public void AddProceduralTexture(string name, int width, int height, 
        Func<int, int, int, int, uint> generator, string category = "Procedural")
    {
        var pixels = new uint[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                pixels[y * width + x] = generator(x, y, width, height);
            }
        }

        AddTexture(name, pixels, width, height, category);
    }

    /// <summary>
    /// Train MultiScaleContextGraph on texture for ultra-realistic synthesis
    /// </summary>
    public void TrainTexture(int textureId, int quantizationLevel = 128, int scaleRadius = 4)
    {
        var texture = _textures.FirstOrDefault(t => t.Id == textureId);
        if (texture == null)
            return;

        Console.WriteLine($"[TextureLibrary] Training texture '{texture.Name}' with {texture.Pixels.Length} pixels");

        // Initialize graph and quantizer
        texture.TrainedGraph = new MultiScaleContextGraph();
        texture.TrainedGraph.SetGpuAccelerator(_gpu);

        // Build color quantizer
        var colors = texture.Pixels.Select(p => new ColorRgb(
            (byte)((p >> 16) & 0xFF),
            (byte)((p >> 8) & 0xFF),
            (byte)(p & 0xFF)
        )).ToList();

        texture.Quantizer = new ColorQuantizer(quantizationLevel, _gpu);
        texture.Quantizer.BuildPalette(colors);

        Console.WriteLine($"[TextureLibrary] Quantizer built with {quantizationLevel} colors");

        // Train graph on texture patterns
        var quantizedPixels = new ColorRgb[texture.Pixels.Length];
        for (int i = 0; i < texture.Pixels.Length; i++)
        {
            var pixel = texture.Pixels[i];
            var color = new ColorRgb(
                (byte)((pixel >> 16) & 0xFF),
                (byte)((pixel >> 8) & 0xFF),
                (byte)(pixel & 0xFF)
            );
            quantizedPixels[i] = texture.Quantizer.Quantize(color);
        }

        // Extract patterns from texture
        int patternsExtracted = 0;
        for (int y = 0; y < texture.Height; y++)
        {
            for (int x = 0; x < texture.Width; x++)
            {
                int idx = y * texture.Width + x;
                var centerColor = quantizedPixels[idx];

                float normalizedX = texture.Width > 1 ? (float)x / (texture.Width - 1) : 0.5f;
                float normalizedY = texture.Height > 1 ? (float)y / (texture.Height - 1) : 0.5f;

                // Extract 3x3 neighborhood
                for (int dir = 0; dir < 8; dir++)
                {
                    var direction = (Direction)dir;
                    var (dx, dy) = direction.GetOffset();
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < texture.Width && ny >= 0 && ny < texture.Height)
                    {
                        int nidx = ny * texture.Width + nx;
                        var targetColor = quantizedPixels[nidx];

                        var neighbors = new Dictionary<Direction, ColorRgb?>();
                        for (int d = 0; d < 8; d++)
                        {
                            var dir2 = (Direction)d;
                            var (dx2, dy2) = dir2.GetOffset();
                            int nx2 = x + dx2;
                            int ny2 = y + dy2;

                            if (nx2 >= 0 && nx2 < texture.Width && ny2 >= 0 && ny2 < texture.Height)
                            {
                                neighbors[dir2] = quantizedPixels[ny2 * texture.Width + nx2];
                            }
                            else
                            {
                                neighbors[dir2] = null;
                            }
                        }

                        texture.TrainedGraph.AddPatternMultiScale(
                            centerColor, neighbors, direction, targetColor,
                            texture.Pixels, texture.Width, texture.Height, x, y
                        );

                        patternsExtracted++;
                    }
                }
            }
        }

        texture.TrainedGraph.Normalize();

        Console.WriteLine($"[TextureLibrary] Training complete: {patternsExtracted} patterns extracted");
        Console.WriteLine($"[TextureLibrary] Graph contains {texture.TrainedGraph.GetTotalPatternCount()} unique patterns");
    }

    /// <summary>
    /// Synthesize new texture using trained AI model
    /// </summary>
    public uint[] SynthesizeTexture(int textureId, int width, int height, int seedCount = 4)
    {
        var texture = _textures.FirstOrDefault(t => t.Id == textureId);
        if (texture?.TrainedGraph == null)
            return GenerateFallbackTexture(width, height);

        Console.WriteLine($"[TextureLibrary] Synthesizing {width}x{height} texture from '{texture.Name}'");

        var fastGraph = texture.TrainedGraph.GetFastGraph();
        var structuralGraph = texture.TrainedGraph.GetStructuralGraph();

        var wfc = new Services.FastWaveFunctionCollapse(
            fastGraph, width, height, _gpu, 0.0, 
            System.Threading.CancellationToken.None, structuralGraph
        );

        var pixels = wfc.Generate(seedCount, Services.FastWaveFunctionCollapse.GenerationMode.FastOrganic, null, 1, false);

        Console.WriteLine($"[TextureLibrary] Texture synthesis complete");
        return pixels;
    }

    public TextureEntry? GetTexture(int id) => _textures.FirstOrDefault(t => t.Id == id);

    public List<string> GetCategories() => _textures.Select(t => t.Category).Distinct().ToList();

    public List<TextureEntry> GetTexturesByCategory(string category) =>
        _textures.Where(t => t.Category == category).ToList();

    // Procedural texture generators
    private static uint GenerateGrassTexture(int x, int y, int w, int h)
    {
        float noise = Noise(x * 0.1f, y * 0.1f);
        byte g = (byte)(80 + noise * 40);
        byte r = (byte)(g * 0.5f);
        byte b = (byte)(g * 0.3f);
        return (uint)((255 << 24) | (r << 16) | (g << 8) | b);
    }

    private static uint GenerateRockTexture(int x, int y, int w, int h)
    {
        float noise = Noise(x * 0.05f, y * 0.05f);
        byte gray = (byte)(90 + noise * 60);
        return (uint)((255 << 24) | (gray << 16) | (gray << 8) | gray);
    }

    private static uint GenerateSandTexture(int x, int y, int w, int h)
    {
        float noise = Noise(x * 0.2f, y * 0.2f);
        byte r = (byte)(220 + noise * 35);
        byte g = (byte)(200 + noise * 30);
        byte b = (byte)(160 + noise * 20);
        return (uint)((255 << 24) | (r << 16) | (g << 8) | b);
    }

    private static uint GenerateSnowTexture(int x, int y, int w, int h)
    {
        float noise = Noise(x * 0.3f, y * 0.3f);
        byte val = (byte)(240 + noise * 15);
        return (uint)((255 << 24) | (val << 16) | (val << 8) | 255);
    }

    private static uint GenerateDirtTexture(int x, int y, int w, int h)
    {
        float noise = Noise(x * 0.15f, y * 0.15f);
        byte r = (byte)(100 + noise * 40);
        byte g = (byte)(70 + noise * 30);
        byte b = (byte)(40 + noise * 20);
        return (uint)((255 << 24) | (r << 16) | (g << 8) | b);
    }

    private static uint GenerateStoneTexture(int x, int y, int w, int h)
    {
        float noise1 = Noise(x * 0.08f, y * 0.08f);
        float noise2 = Noise(x * 0.2f, y * 0.2f) * 0.3f;
        byte gray = (byte)(120 + (noise1 + noise2) * 50);
        return (uint)((255 << 24) | (gray << 16) | (gray << 8) | gray);
    }

    private static uint[] GenerateFallbackTexture(int width, int height)
    {
        var pixels = new uint[width * height];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = 0xFF808080; // Gray
        }
        return pixels;
    }

    private static float Noise(float x, float y)
    {
        int n = (int)(x + y * 57);
        n = (n << 13) ^ n;
        return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f) * 0.5f + 0.5f;
    }
}
