using System;
using System.Linq;

namespace NNImage.Models;

/// <summary>
/// Procedural heightmap generator using multiple noise algorithms
/// Generates realistic terrain elevations with configurable parameters
/// </summary>
public class HeightmapGenerator
{
    private readonly Random _random;
    private int _seed;

    public HeightmapGenerator(int seed = 0)
    {
        _seed = seed == 0 ? Random.Shared.Next() : seed;
        _random = new Random(_seed);
    }

    public enum TerrainType
    {
        Mountains,
        Hills,
        Plains,
        Islands,
        Canyons,
        Valleys,
        Plateaus,
        Volcanic,
        Rolling,
        Alpine
    }

    /// <summary>
    /// Generate heightmap with specified parameters
    /// Returns array of heights normalized to 0.0-1.0
    /// </summary>
    public float[] Generate(int width, int height, TerrainType terrainType, 
        float roughness = 0.5f, int octaves = 6, float scale = 100f)
    {
        Console.WriteLine($"[HeightmapGenerator] Generating {terrainType} terrain");
        Console.WriteLine($"  Size: {width}x{height}, Scale: {scale}, Octaves: {octaves}, Roughness: {roughness}");
        Console.WriteLine($"  Seed: {_seed}");

        var heightmap = new float[width * height];

        // Generate base terrain based on type
        switch (terrainType)
        {
            case TerrainType.Mountains:
                GenerateMountains(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Hills:
                GenerateHills(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Plains:
                GeneratePlains(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Islands:
                GenerateIslands(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Canyons:
                GenerateCanyons(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Valleys:
                GenerateValleys(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Plateaus:
                GeneratePlateaus(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Volcanic:
                GenerateVolcanic(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Rolling:
                GenerateRolling(heightmap, width, height, roughness, octaves, scale);
                break;
            case TerrainType.Alpine:
                GenerateAlpine(heightmap, width, height, roughness, octaves, scale);
                break;
        }

        // Normalize to 0-1 range
        NormalizeHeightmap(heightmap);

        return heightmap;
    }

    private void GenerateMountains(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Multi-octave Perlin noise for mountain ranges
        Console.WriteLine($"[GenerateMountains] Processing {width}x{height} pixels...");

        float testNoise1 = PerlinNoise(0f, 0f);
        float testNoise2 = PerlinNoise(1f, 1f);
        float testNoise3 = PerlinNoise(10f, 10f);
        Console.WriteLine($"[GenerateMountains] Test noise values: {testNoise1:F3}, {testNoise2:F3}, {testNoise3:F3}");

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                float elevation = 0f;
                float amplitude = 1f;
                float frequency = 1f;
                float maxValue = 0f;

                for (int o = 0; o < octaves; o++)
                {
                    float sample = PerlinNoise(nx * frequency + _seed, ny * frequency + _seed);
                    elevation += sample * amplitude;
                    maxValue += amplitude;

                    amplitude *= roughness;
                    frequency *= 2f;
                }

                elevation /= maxValue;

                // Apply mountain shaping - emphasize peaks
                elevation = MathF.Pow(elevation, 0.7f);

                heightmap[y * width + x] = elevation;
            }
        }

        // Debug: check first few values
        Console.WriteLine($"[GenerateMountains] First 10 values: {string.Join(", ", heightmap.Take(10).Select(h => h.ToString("F3")))}");
    }

    private void GenerateHills(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Gentler terrain with rolling hills
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / (scale * 1.5f);
                float ny = y / (scale * 1.5f);

                float elevation = FractalNoise(nx, ny, octaves, roughness);
                elevation = MathF.Pow(elevation, 1.2f); // Softer peaks

                heightmap[y * width + x] = elevation * 0.6f; // Lower maximum height
            }
        }
    }

    private void GeneratePlains(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Very flat terrain with subtle variations
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / (scale * 3f);
                float ny = y / (scale * 3f);

                float elevation = FractalNoise(nx, ny, 3, 0.3f);
                elevation = MathF.Pow(elevation, 2.0f); // Very gentle

                heightmap[y * width + x] = elevation * 0.3f; // Very low height variation
            }
        }
    }

    private void GenerateIslands(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Island with water around edges
        float centerX = width / 2f;
        float centerY = height / 2f;
        float maxDist = MathF.Min(width, height) / 2f;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                // Base terrain
                float elevation = FractalNoise(nx, ny, octaves, roughness);

                // Distance from center for island falloff
                float dx = x - centerX;
                float dy = y - centerY;
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                float falloff = 1f - MathF.Pow(dist / maxDist, 2f);
                falloff = MathF.Max(0, falloff);

                elevation *= falloff;

                heightmap[y * width + x] = elevation;
            }
        }
    }

    private void GenerateCanyons(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Deep valleys and high plateaus
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                // Create ridges using absolute value
                float elevation = MathF.Abs(FractalNoise(nx, ny, octaves, roughness) * 2f - 1f);
                elevation = 1f - elevation; // Invert to create valleys
                elevation = MathF.Pow(elevation, 1.5f); // Sharpen canyon walls

                heightmap[y * width + x] = elevation;
            }
        }
    }

    private void GenerateValleys(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // River valleys with surrounding hills
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                // Base terrain
                float elevation = FractalNoise(nx, ny, octaves, roughness);

                // Add valley carving
                float valleyNoise = PerlinNoise(nx * 0.3f + _seed, ny * 0.3f + _seed);
                float valley = MathF.Abs(valleyNoise * 2f - 1f);
                valley = MathF.Pow(valley, 2f);

                elevation *= valley;

                heightmap[y * width + x] = elevation;
            }
        }
    }

    private void GeneratePlateaus(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Flat-topped highlands
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                float elevation = FractalNoise(nx, ny, octaves, roughness);

                // Create plateau effect with step function
                if (elevation > 0.6f)
                    elevation = 0.8f + (elevation - 0.6f) * 0.5f;
                else if (elevation > 0.4f)
                    elevation = 0.6f + (elevation - 0.4f);

                heightmap[y * width + x] = elevation;
            }
        }
    }

    private void GenerateVolcanic(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Volcanic peaks with calderas
        float centerX = width / 2f;
        float centerY = height / 2f;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / scale;
                float ny = y / scale;

                // Distance from center
                float dx = (x - centerX) / width;
                float dy = (y - centerY) / height;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                // Volcanic cone shape
                float cone = 1f - dist * 2f;
                cone = MathF.Max(0, cone);

                // Add crater at top
                if (cone > 0.7f)
                    cone = 0.7f - (cone - 0.7f) * 3f;

                // Add noise for natural look
                float noise = FractalNoise(nx, ny, octaves, roughness) * 0.3f;

                heightmap[y * width + x] = MathF.Max(0, cone + noise);
            }
        }
    }

    private void GenerateRolling(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Smooth rolling terrain
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / (scale * 2f);
                float ny = y / (scale * 2f);

                // Use sine waves for smooth rolling
                float wave1 = MathF.Sin(nx * 3f) * 0.5f + 0.5f;
                float wave2 = MathF.Sin(ny * 3f) * 0.5f + 0.5f;
                float waves = (wave1 + wave2) * 0.5f;

                // Add fractal noise for detail
                float noise = FractalNoise(nx * 2f, ny * 2f, octaves, roughness * 0.5f);

                float elevation = waves * 0.7f + noise * 0.3f;

                heightmap[y * width + x] = elevation * 0.5f;
            }
        }
    }

    private void GenerateAlpine(float[] heightmap, int width, int height, 
        float roughness, int octaves, float scale)
    {
        // Jagged alpine peaks
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float nx = x / (scale * 0.8f);
                float ny = y / (scale * 0.8f);

                // Sharp peaks using ridge noise
                float ridge = MathF.Abs(FractalNoise(nx, ny, octaves, roughness) * 2f - 1f);
                ridge = 1f - ridge;
                ridge = MathF.Pow(ridge, 0.5f); // Sharpen peaks

                // Add detail
                float detail = FractalNoise(nx * 4f, ny * 4f, 4, 0.5f) * 0.2f;

                heightmap[y * width + x] = MathF.Min(1f, ridge + detail);
            }
        }
    }

    private float FractalNoise(float x, float y, int octaves, float persistence)
    {
        float total = 0f;
        float frequency = 1f;
        float amplitude = 1f;
        float maxValue = 0f;

        for (int i = 0; i < octaves; i++)
        {
            total += PerlinNoise(x * frequency + _seed, y * frequency + _seed) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2f;
        }

        return total / maxValue;
    }

    private void NormalizeHeightmap(float[] heightmap)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (float h in heightmap)
        {
            if (h < min) min = h;
            if (h > max) max = h;
        }

        float range = max - min;
        if (range > 0)
        {
            for (int i = 0; i < heightmap.Length; i++)
            {
                heightmap[i] = (heightmap[i] - min) / range;
            }
        }
    }

    /// <summary>
    /// Perlin noise implementation
    /// </summary>
    private float PerlinNoise(float x, float y)
    {
        // Get integer parts
        int xi = (int)MathF.Floor(x);
        int yi = (int)MathF.Floor(y);

        // Get fractional parts
        float xf = x - xi;
        float yf = y - yi;

        // Wrap to 0-255
        xi = xi & 255;
        yi = yi & 255;

        // Fade curves
        float u = Fade(xf);
        float v = Fade(yf);

        // Hash coordinates of the 4 corners
        int aa = Permutation[(Permutation[xi] + yi) & 255];
        int ab = Permutation[(Permutation[xi] + yi + 1) & 255];
        int ba = Permutation[(Permutation[xi + 1] + yi) & 255];
        int bb = Permutation[(Permutation[xi + 1] + yi + 1) & 255];

        // Blend results from corners
        float x1 = Lerp(Grad(aa, xf, yf), Grad(ba, xf - 1, yf), u);
        float x2 = Lerp(Grad(ab, xf, yf - 1), Grad(bb, xf - 1, yf - 1), u);

        float result = Lerp(x1, x2, v);

        // Map from [-1, 1] to [0, 1]
        return (result + 1.0f) * 0.5f;
    }

    private float Fade(float t) => t * t * t * (t * (t * 6 - 15) + 10);
    private float Lerp(float a, float b, float t) => a + t * (b - a);

    private float Grad(int hash, float x, float y)
    {
        // Convert hash to one of 8 gradient directions
        int h = hash & 7;
        float u = h < 4 ? x : y;
        float v = h < 4 ? y : x;
        return ((h & 1) != 0 ? -u : u) + ((h & 2) != 0 ? -v : v);
    }

    private static readonly int[] Permutation = GeneratePermutation();

    private static int[] GeneratePermutation()
    {
        var p = new int[512];
        var perm = new int[256];

        for (int i = 0; i < 256; i++)
            perm[i] = i;

        // Shuffle
        var rng = new Random(12345); // Fixed seed for consistency
        for (int i = 255; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }

        for (int i = 0; i < 512; i++)
            p[i] = perm[i % 256];

        return p;
    }
}
