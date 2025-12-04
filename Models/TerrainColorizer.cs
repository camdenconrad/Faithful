using System;

namespace NNImage.Models;

/// <summary>
/// Ultra-realistic terrain colorization based on elevation, slope, and environmental factors
/// </summary>
public class TerrainColorizer
{
    public enum ColorScheme
    {
        Realistic,
        Fantasy,
        Desert,
        Arctic,
        Tropical,
        Alien,
        Autumn,
        Monochrome
    }

    /// <summary>
    /// Colorize heightmap with realistic terrain colors
    /// Returns ARGB pixel array
    /// </summary>
    public uint[] Colorize(float[] heightmap, int width, int height, 
        ColorScheme scheme, bool useLighting = true, float sunAngle = 45f)
    {
        var pixels = new uint[width * height];
        var normals = useLighting ? CalculateNormals(heightmap, width, height) : null;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                float elevation = heightmap[idx];
                float slope = CalculateSlope(heightmap, width, height, x, y);

                // Get base color for elevation and slope
                var color = GetTerrainColor(elevation, slope, scheme);

                // Apply lighting if enabled
                if (useLighting && normals != null)
                {
                    var normal = normals[idx];
                    float light = CalculateLighting(normal, sunAngle);
                    color = ApplyLighting(color, light);
                }

                pixels[idx] = ColorToUint(color);
            }
        }

        return pixels;
    }

    private (byte r, byte g, byte b) GetTerrainColor(float elevation, float slope, ColorScheme scheme)
    {
        return scheme switch
        {
            ColorScheme.Realistic => GetRealisticColor(elevation, slope),
            ColorScheme.Fantasy => GetFantasyColor(elevation, slope),
            ColorScheme.Desert => GetDesertColor(elevation, slope),
            ColorScheme.Arctic => GetArcticColor(elevation, slope),
            ColorScheme.Tropical => GetTropicalColor(elevation, slope),
            ColorScheme.Alien => GetAlienColor(elevation, slope),
            ColorScheme.Autumn => GetAutumnColor(elevation, slope),
            ColorScheme.Monochrome => GetMonochromeColor(elevation, slope),
            _ => GetRealisticColor(elevation, slope)
        };
    }

    private (byte r, byte g, byte b) GetRealisticColor(float elevation, float slope)
    {
        // Ultra-realistic Earth-like terrain colors

        // Deep water (dark blue)
        if (elevation < 0.3f)
        {
            float depth = 0.3f - elevation;
            byte b = (byte)Math.Clamp(100 + depth * 500, 0, 255);
            return (30, 60, b);
        }

        // Shallow water/beach transition (cyan to sand)
        if (elevation < 0.35f)
        {
            float t = (elevation - 0.3f) / 0.05f;
            return LerpColor((30, 144, 255), (238, 214, 175), t);
        }

        // Beach (sand)
        if (elevation < 0.38f)
        {
            return (238, 214, 175);
        }

        // Coastal grass
        if (elevation < 0.42f)
        {
            float t = (elevation - 0.38f) / 0.04f;
            return LerpColor((238, 214, 175), (124, 176, 85), t);
        }

        // Grassland/lowland
        if (elevation < 0.55f)
        {
            // Vary grass color based on slope
            if (slope > 0.4f)
                return (139, 137, 112); // Dry grass on slopes
            return (124, 176, 85); // Rich green grass
        }

        // Forest (dark green)
        if (elevation < 0.65f)
        {
            if (slope > 0.5f)
                return (105, 105, 90); // Rocky slopes
            return (34, 139, 34); // Forest green
        }

        // Rocky mountain slopes
        if (elevation < 0.75f)
        {
            if (slope > 0.6f)
                return (128, 128, 128); // Steep rocky cliffs
            return (139, 137, 112); // Alpine meadows
        }

        // High alpine / sparse vegetation
        if (elevation < 0.82f)
        {
            float t = (elevation - 0.75f) / 0.07f;
            return LerpColor((139, 137, 112), (180, 180, 180), t);
        }

        // Snow line transition
        if (elevation < 0.87f)
        {
            float t = (elevation - 0.82f) / 0.05f;
            return LerpColor((180, 180, 180), (240, 240, 255), t);
        }

        // Snow-capped peaks
        return (250, 250, 255);
    }

    private (byte r, byte g, byte b) GetFantasyColor(float elevation, float slope)
    {
        // Vibrant fantasy colors
        if (elevation < 0.3f) return (138, 43, 226); // Purple water
        if (elevation < 0.35f) return (255, 20, 147); // Pink shores
        if (elevation < 0.5f) return (0, 255, 127); // Spring green
        if (elevation < 0.65f) return (64, 224, 208); // Turquoise forests
        if (elevation < 0.8f) return (255, 140, 0); // Orange mountains
        return (255, 215, 0); // Golden peaks
    }

    private (byte r, byte g, byte b) GetDesertColor(float elevation, float slope)
    {
        // Desert terrain
        if (elevation < 0.3f) return (85, 107, 47); // Dark olive (oasis water)
        if (elevation < 0.35f) return (189, 183, 107); // Dark khaki (wet sand)
        if (elevation < 0.5f) return (237, 201, 175); // Light sand
        if (elevation < 0.7f) return (210, 180, 140); // Tan dunes
        if (elevation < 0.85f) return (160, 82, 45); // Sienna rock
        return (139, 69, 19); // Saddle brown (peaks)
    }

    private (byte r, byte g, byte b) GetArcticColor(float elevation, float slope)
    {
        // Arctic/frozen terrain
        if (elevation < 0.3f) return (25, 25, 112); // Midnight blue (deep water)
        if (elevation < 0.35f) return (70, 130, 180); // Steel blue (ice edge)
        if (elevation < 0.5f) return (176, 224, 230); // Powder blue (ice shelf)
        if (elevation < 0.7f) return (230, 230, 250); // Lavender (snow)
        if (elevation < 0.85f) return (240, 248, 255); // Alice blue
        return (255, 250, 250); // Snow white (peaks)
    }

    private (byte r, byte g, byte b) GetTropicalColor(float elevation, float slope)
    {
        // Lush tropical terrain
        if (elevation < 0.3f) return (0, 191, 255); // Deep sky blue (water)
        if (elevation < 0.35f) return (240, 230, 140); // Khaki (beach)
        if (elevation < 0.5f) return (60, 179, 113); // Medium sea green
        if (elevation < 0.7f) return (0, 100, 0); // Dark green (jungle)
        if (elevation < 0.85f) return (107, 142, 35); // Olive drab (highlands)
        return (139, 137, 112); // Dark khaki (peaks)
    }

    private (byte r, byte g, byte b) GetAlienColor(float elevation, float slope)
    {
        // Alien world colors
        if (elevation < 0.3f) return (255, 0, 255); // Magenta liquid
        if (elevation < 0.35f) return (0, 255, 255); // Cyan shores
        if (elevation < 0.5f) return (255, 255, 0); // Yellow lowlands
        if (elevation < 0.7f) return (0, 255, 0); // Green mid-level
        if (elevation < 0.85f) return (255, 0, 0); // Red highlands
        return (128, 0, 128); // Purple peaks
    }

    private (byte r, byte g, byte b) GetAutumnColor(float elevation, float slope)
    {
        // Autumn forest colors
        if (elevation < 0.3f) return (70, 130, 180); // Steel blue (water)
        if (elevation < 0.35f) return (210, 180, 140); // Tan (shore)
        if (elevation < 0.5f) return (218, 165, 32); // Goldenrod
        if (elevation < 0.65f) return (255, 140, 0); // Dark orange
        if (elevation < 0.8f) return (178, 34, 34); // Firebrick
        return (139, 69, 19); // Saddle brown
    }

    private (byte r, byte g, byte b) GetMonochromeColor(float elevation, float slope)
    {
        // Grayscale heightmap visualization
        byte gray = (byte)(elevation * 255);
        return (gray, gray, gray);
    }

    private (byte r, byte g, byte b) LerpColor((byte r, byte g, byte b) a, (byte r, byte g, byte b) b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return (
            (byte)(a.r + (b.r - a.r) * t),
            (byte)(a.g + (b.g - a.g) * t),
            (byte)(a.b + (b.b - a.b) * t)
        );
    }

    private float CalculateSlope(float[] heightmap, int width, int height, int x, int y)
    {
        float h = heightmap[y * width + x];
        float hx = x < width - 1 ? heightmap[y * width + x + 1] : h;
        float hy = y < height - 1 ? heightmap[(y + 1) * width + x] : h;

        float dx = hx - h;
        float dy = hy - h;

        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private (float x, float y, float z)[] CalculateNormals(float[] heightmap, int width, int height)
    {
        var normals = new (float, float, float)[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Sample neighbors for normal calculation
                float left = x > 0 ? heightmap[y * width + x - 1] : heightmap[y * width + x];
                float right = x < width - 1 ? heightmap[y * width + x + 1] : heightmap[y * width + x];
                float top = y > 0 ? heightmap[(y - 1) * width + x] : heightmap[y * width + x];
                float bottom = y < height - 1 ? heightmap[(y + 1) * width + x] : heightmap[y * width + x];

                // Calculate normal vector
                float dx = (right - left) * 2f;
                float dy = (bottom - top) * 2f;
                float dz = 1f;

                // Normalize
                float len = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                normals[y * width + x] = (dx / len, dy / len, dz / len);
            }
        }

        return normals;
    }

    private float CalculateLighting((float x, float y, float z) normal, float sunAngleDegrees)
    {
        // Sun direction vector
        float angleRad = sunAngleDegrees * MathF.PI / 180f;
        float sunX = MathF.Cos(angleRad);
        float sunY = 0f;
        float sunZ = MathF.Sin(angleRad);

        // Dot product for lighting (Lambertian reflection)
        float dot = normal.x * sunX + normal.y * sunY + normal.z * sunZ;

        // Ambient + diffuse lighting
        float ambient = 0.3f;
        float diffuse = Math.Max(0, dot) * 0.7f;

        return ambient + diffuse;
    }

    private (byte r, byte g, byte b) ApplyLighting((byte r, byte g, byte b) color, float light)
    {
        return (
            (byte)Math.Clamp(color.r * light, 0, 255),
            (byte)Math.Clamp(color.g * light, 0, 255),
            (byte)Math.Clamp(color.b * light, 0, 255)
        );
    }

    private uint ColorToUint((byte r, byte g, byte b) color)
    {
        return (uint)((255 << 24) | (color.r << 16) | (color.g << 8) | color.b);
    }
}
