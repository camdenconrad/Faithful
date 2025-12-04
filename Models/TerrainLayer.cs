using System;

namespace NNImage.Models;

/// <summary>
/// Layer system for non-destructive terrain editing
/// </summary>
public class TerrainLayer
{
    public enum LayerType
    {
        Height,
        Texture,
        Mask,
        Detail
    }

    public int Id { get; set; }
    public string Name { get; set; } = "Layer";
    public LayerType Type { get; set; } = LayerType.Height;
    public bool Visible { get; set; } = true;
    public float Opacity { get; set; } = 1.0f;
    public BlendMode Blend { get; set; } = BlendMode.Normal;

    // Layer data
    public float[]? HeightData { get; set; }
    public byte[]? TextureData { get; set; }
    public byte[]? MaskData { get; set; }

    public int Width { get; set; }
    public int Height { get; set; }

    public enum BlendMode
    {
        Normal,
        Add,
        Subtract,
        Multiply,
        Overlay,
        Screen
    }

    public TerrainLayer(int width, int height, LayerType type = LayerType.Height)
    {
        Width = width;
        Height = height;
        Type = type;

        switch (type)
        {
            case LayerType.Height:
            case LayerType.Detail:
                HeightData = new float[width * height];
                break;
            case LayerType.Texture:
                TextureData = new byte[width * height];
                break;
            case LayerType.Mask:
                MaskData = new byte[width * height];
                Array.Fill(MaskData, (byte)255); // Fully visible by default
                break;
        }
    }

    public TerrainLayer Clone()
    {
        var clone = new TerrainLayer(Width, Height, Type)
        {
            Id = Id,
            Name = Name + " (Copy)",
            Visible = Visible,
            Opacity = Opacity,
            Blend = Blend
        };

        if (HeightData != null)
        {
            clone.HeightData = new float[HeightData.Length];
            Array.Copy(HeightData, clone.HeightData, HeightData.Length);
        }

        if (TextureData != null)
        {
            clone.TextureData = new byte[TextureData.Length];
            Array.Copy(TextureData, clone.TextureData, TextureData.Length);
        }

        if (MaskData != null)
        {
            clone.MaskData = new byte[MaskData.Length];
            Array.Copy(MaskData, clone.MaskData, MaskData.Length);
        }

        return clone;
    }

    public void Clear()
    {
        if (HeightData != null)
            Array.Fill(HeightData, 0f);
        if (TextureData != null)
            Array.Fill(TextureData, (byte)0);
        if (MaskData != null)
            Array.Fill(MaskData, (byte)255);
    }
}

/// <summary>
/// Manages multiple terrain layers and compositing
/// </summary>
public class LayerStack
{
    private readonly System.Collections.Generic.List<TerrainLayer> _layers = new();
    private int _nextId = 1;

    public System.Collections.Generic.IReadOnlyList<TerrainLayer> Layers => _layers;
    public int Width { get; private set; }
    public int Height { get; private set; }

    public LayerStack(int width, int height)
    {
        Width = width;
        Height = height;

        // Create base layer
        AddLayer("Base Height", TerrainLayer.LayerType.Height);
        AddLayer("Base Texture", TerrainLayer.LayerType.Texture);
    }

    public TerrainLayer AddLayer(string name, TerrainLayer.LayerType type)
    {
        var layer = new TerrainLayer(Width, Height, type)
        {
            Id = _nextId++,
            Name = name
        };

        _layers.Add(layer);
        return layer;
    }

    public void RemoveLayer(int layerId)
    {
        var layer = _layers.Find(l => l.Id == layerId);
        if (layer != null && _layers.Count > 1) // Keep at least one layer
        {
            _layers.Remove(layer);
        }
    }

    public TerrainLayer? GetLayer(int id) => _layers.Find(l => l.Id == id);

    public void MoveLayer(int layerId, int newIndex)
    {
        var layer = _layers.Find(l => l.Id == layerId);
        if (layer != null)
        {
            _layers.Remove(layer);
            newIndex = Math.Clamp(newIndex, 0, _layers.Count);
            _layers.Insert(newIndex, layer);
        }
    }

    /// <summary>
    /// Composite all visible layers into final heightmap
    /// </summary>
    public float[] CompositeHeightLayers()
    {
        var result = new float[Width * Height];

        foreach (var layer in _layers)
        {
            if (!layer.Visible || layer.HeightData == null)
                continue;

            for (int i = 0; i < result.Length; i++)
            {
                float layerValue = layer.HeightData[i];
                float mask = 1.0f; // TODO: Apply mask layers

                float blended = BlendHeight(result[i], layerValue, layer.Blend);
                result[i] = result[i] + (blended - result[i]) * layer.Opacity * mask;
            }
        }

        return result;
    }

    /// <summary>
    /// Composite all visible texture layers
    /// </summary>
    public byte[] CompositeTextureLayers()
    {
        var result = new byte[Width * Height];

        foreach (var layer in _layers)
        {
            if (!layer.Visible || layer.TextureData == null)
                continue;

            for (int i = 0; i < result.Length; i++)
            {
                if (layer.Opacity > 0.5f) // Simple blending for texture IDs
                {
                    result[i] = layer.TextureData[i];
                }
            }
        }

        return result;
    }

    private float BlendHeight(float baseVal, float layerVal, TerrainLayer.BlendMode mode)
    {
        return mode switch
        {
            TerrainLayer.BlendMode.Normal => layerVal,
            TerrainLayer.BlendMode.Add => Math.Clamp(baseVal + layerVal, 0f, 1f),
            TerrainLayer.BlendMode.Subtract => Math.Clamp(baseVal - layerVal, 0f, 1f),
            TerrainLayer.BlendMode.Multiply => baseVal * layerVal,
            TerrainLayer.BlendMode.Overlay => baseVal < 0.5f
                ? 2f * baseVal * layerVal
                : 1f - 2f * (1f - baseVal) * (1f - layerVal),
            TerrainLayer.BlendMode.Screen => 1f - (1f - baseVal) * (1f - layerVal),
            _ => layerVal
        };
    }
}
