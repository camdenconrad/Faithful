using Avalonia.Media.Imaging;
using System;
using System.Collections.Generic;
using System.IO;

namespace NNImage;

public class ImageTrainer
{
    private readonly AdjacencyGraph _adjacencyGraph = new();
    private readonly ColorQuantizer _quantizer;
    private readonly int _quantizationLevel;
    private bool _paletteBuilt;

    public ImageTrainer(int quantizationLevel)
    {
        _quantizationLevel = quantizationLevel;
        _quantizer = new ColorQuantizer(quantizationLevel);
    }

    public void ProcessImage(string imagePath)
    {
        using var stream = File.OpenRead(imagePath);
        var bitmap = new Bitmap(stream);

        var width = bitmap.PixelSize.Width;
        var height = bitmap.PixelSize.Height;

        // Extract all colors first if palette not built
        if (!_paletteBuilt)
        {
            var allColors = new List<ColorRgb>();

            unsafe
            {
                using var lockedBitmap = bitmap.Lock();
                var ptr = (uint*)lockedBitmap.Address.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = ptr[y * width + x];
                        var color = PixelToColor(pixel);
                        allColors.Add(color);
                    }
                }
            }

            _quantizer.BuildPalette(allColors);
            _paletteBuilt = true;
        }

        // Extract adjacency patterns
        unsafe
        {
            using var lockedBitmap = bitmap.Lock();
            var ptr = (uint*)lockedBitmap.Address.ToPointer();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var centerPixel = ptr[y * width + x];
                    var centerColor = _quantizer.Quantize(PixelToColor(centerPixel));

                    // Check all 8 directions
                    foreach (var direction in DirectionExtensions.AllDirections)
                    {
                        var (dx, dy) = direction.GetOffset();
                        var nx = x + dx;
                        var ny = y + dy;

                        // Check bounds
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            var neighborPixel = ptr[ny * width + nx];
                            var neighborColor = _quantizer.Quantize(PixelToColor(neighborPixel));

                            _adjacencyGraph.AddAdjacency(centerColor, direction, neighborColor);
                        }
                    }
                }
            }
        }
    }

    public AdjacencyGraph GetAdjacencyGraph()
    {
        _adjacencyGraph.Normalize();
        return _adjacencyGraph;
    }

    private ColorRgb PixelToColor(uint pixel)
    {
        var a = (byte)((pixel >> 24) & 0xFF);
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);

        // Handle transparency by blending with white
        if (a < 255)
        {
            var alpha = a / 255.0;
            r = (byte)(r * alpha + 255 * (1 - alpha));
            g = (byte)(g * alpha + 255 * (1 - alpha));
            b = (byte)(b * alpha + 255 * (1 - alpha));
        }

        return new ColorRgb(r, g, b);
    }
}
