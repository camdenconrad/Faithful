using System;

namespace NNImage.Models;

/// <summary>
/// Advanced brush system for terrain editing with various brush types
/// </summary>
public class TerrainBrush
{
    public enum BrushType
    {
        Height,          // Direct height modification
        Smooth,          // Smoothing brush
        Flatten,         // Flatten to target height
        Noise,           // Add procedural noise
        Texture,         // Paint texture
        Erosion,         // Simulate erosion
        Raise,           // Only raise terrain
        Lower,           // Only lower terrain
        Plateau,         // Create flat plateaus
        Cliff            // Create steep cliffs
    }

    public enum BrushShape
    {
        Circle,
        Square,
        Diamond,
        Organic
    }

    public BrushType Type { get; set; } = BrushType.Height;
    public BrushShape Shape { get; set; } = BrushShape.Circle;
    public float Size { get; set; } = 32f;
    public float Strength { get; set; } = 0.5f;
    public float Hardness { get; set; } = 0.5f;
    public int TextureId { get; set; } = 0;
    public float TargetHeight { get; set; } = 0.5f;
    public bool Additive { get; set; } = true;

    /// <summary>
    /// Apply brush stroke at specified position
    /// </summary>
    public void ApplyStroke(float[] heightmap, int width, int height, int x, int y, 
        byte[]? textureMask = null, float deltaTime = 1f)
    {
        int radius = (int)(Size / 2f);
        float strengthDelta = Strength * deltaTime;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int px = x + dx;
                int py = y + dy;

                if (px < 0 || px >= width || py < 0 || py >= height)
                    continue;

                float falloff = CalculateFalloff(dx, dy, radius);
                if (falloff <= 0f)
                    continue;

                int idx = py * width + px;
                float currentHeight = heightmap[idx];

                switch (Type)
                {
                    case BrushType.Height:
                        ApplyHeight(heightmap, idx, falloff, strengthDelta, currentHeight);
                        break;
                    case BrushType.Smooth:
                        ApplySmooth(heightmap, width, height, px, py, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Flatten:
                        ApplyFlatten(heightmap, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Noise:
                        ApplyNoise(heightmap, idx, falloff, strengthDelta, px, py);
                        break;
                    case BrushType.Texture:
                        ApplyTexture(textureMask, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Erosion:
                        ApplyErosion(heightmap, width, height, px, py, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Raise:
                        ApplyRaise(heightmap, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Lower:
                        ApplyLower(heightmap, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Plateau:
                        ApplyPlateau(heightmap, idx, falloff, strengthDelta);
                        break;
                    case BrushType.Cliff:
                        ApplyCliff(heightmap, width, height, px, py, idx, falloff, strengthDelta);
                        break;
                }
            }
        }
    }

    private float CalculateFalloff(int dx, int dy, int radius)
    {
        float distance = MathF.Sqrt(dx * dx + dy * dy);
        float normalizedDist = distance / radius;

        if (normalizedDist >= 1f)
            return 0f;

        // Apply shape
        switch (Shape)
        {
            case BrushShape.Circle:
                break; // Already circular
            case BrushShape.Square:
                normalizedDist = Math.Max(Math.Abs(dx), Math.Abs(dy)) / (float)radius;
                break;
            case BrushShape.Diamond:
                normalizedDist = (Math.Abs(dx) + Math.Abs(dy)) / (float)radius;
                break;
            case BrushShape.Organic:
                // Add noise to edge
                float noise = PerlinNoise(dx * 0.2f, dy * 0.2f) * 0.3f;
                normalizedDist += noise;
                break;
        }

        if (normalizedDist >= 1f)
            return 0f;

        // Apply hardness curve
        float falloff = 1f - normalizedDist;
        falloff = MathF.Pow(falloff, 1f / Math.Max(0.1f, Hardness));

        return falloff;
    }

    private void ApplyHeight(float[] heightmap, int idx, float falloff, float strength, float current)
    {
        float delta = strength * falloff;
        if (Additive)
            heightmap[idx] = Math.Clamp(current + delta, 0f, 1f);
        else
            heightmap[idx] = Math.Clamp(current - delta, 0f, 1f);
    }

    private void ApplySmooth(float[] heightmap, int width, int height, int px, int py, int idx, float falloff, float strength)
    {
        // Average neighboring heights
        float sum = 0f;
        int count = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = px + dx;
                int ny = py + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    sum += heightmap[ny * width + nx];
                    count++;
                }
            }
        }

        float avg = sum / count;
        float current = heightmap[idx];
        heightmap[idx] = current + (avg - current) * strength * falloff;
    }

    private void ApplyFlatten(float[] heightmap, int idx, float falloff, float strength)
    {
        float current = heightmap[idx];
        heightmap[idx] = current + (TargetHeight - current) * strength * falloff;
    }

    private void ApplyNoise(float[] heightmap, int idx, float falloff, float strength, int px, int py)
    {
        float noise = PerlinNoise(px * 0.1f, py * 0.1f) * 2f - 1f;
        heightmap[idx] = Math.Clamp(heightmap[idx] + noise * strength * falloff, 0f, 1f);
    }

    private void ApplyTexture(byte[]? textureMask, int idx, float falloff, float strength)
    {
        if (textureMask == null)
            return;

        byte current = textureMask[idx];
        float blend = strength * falloff;

        // Blend towards target texture
        if (current != TextureId)
        {
            if (blend > 0.5f)
                textureMask[idx] = (byte)TextureId;
        }
    }

    private void ApplyErosion(float[] heightmap, int width, int height, int px, int py, int idx, float falloff, float strength)
    {
        // Simple thermal erosion - move material from high to low
        float current = heightmap[idx];
        float lowest = current;
        int lowestIdx = idx;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0) continue;

                int nx = px + dx;
                int ny = py + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    int nidx = ny * width + nx;
                    if (heightmap[nidx] < lowest)
                    {
                        lowest = heightmap[nidx];
                        lowestIdx = nidx;
                    }
                }
            }
        }

        if (lowestIdx != idx)
        {
            float diff = (current - lowest) * 0.5f;
            float transfer = diff * strength * falloff;
            heightmap[idx] -= transfer;
            heightmap[lowestIdx] += transfer * 0.5f; // Some material is lost
        }
    }

    private void ApplyRaise(float[] heightmap, int idx, float falloff, float strength)
    {
        heightmap[idx] = Math.Clamp(heightmap[idx] + strength * falloff, 0f, 1f);
    }

    private void ApplyLower(float[] heightmap, int idx, float falloff, float strength)
    {
        heightmap[idx] = Math.Clamp(heightmap[idx] - strength * falloff, 0f, 1f);
    }

    private void ApplyPlateau(float[] heightmap, int idx, float falloff, float strength)
    {
        float current = heightmap[idx];
        // Snap to nearest 0.1 elevation
        float snapped = MathF.Round(current * 10f) / 10f;
        heightmap[idx] = current + (snapped - current) * strength * falloff;
    }

    private void ApplyCliff(float[] heightmap, int width, int height, int px, int py, int idx, float falloff, float strength)
    {
        // Create sharp elevation changes
        float current = heightmap[idx];
        float gradient = 0f;

        // Calculate gradient direction
        if (px > 0)
        {
            gradient = heightmap[idx] - heightmap[idx - 1];
        }

        // Amplify gradient
        heightmap[idx] = Math.Clamp(current + gradient * strength * falloff * 2f, 0f, 1f);
    }

    private static float PerlinNoise(float x, float y)
    {
        int xi = (int)MathF.Floor(x) & 255;
        int yi = (int)MathF.Floor(y) & 255;
        float xf = x - MathF.Floor(x);
        float yf = y - MathF.Floor(y);
        float u = xf * xf * (3f - 2f * xf);
        float v = yf * yf * (3f - 2f * yf);

        int a = (xi + yi * 57) * 31;
        int b = (xi + 1 + yi * 57) * 31;
        int c = (xi + (yi + 1) * 57) * 31;
        int d = (xi + 1 + (yi + 1) * 57) * 31;

        float x1 = (a % 1000) / 1000f;
        float x2 = (b % 1000) / 1000f;
        float y1 = (c % 1000) / 1000f;
        float y2 = (d % 1000) / 1000f;

        return x1 * (1 - u) * (1 - v) + x2 * u * (1 - v) + y1 * (1 - u) * v + y2 * u * v;
    }
}
