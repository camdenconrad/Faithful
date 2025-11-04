using System;
using System.Collections.Generic;
using System.Linq;

namespace NNImage;

public class ColorQuantizer
{
    private readonly int _colorCount;
    private List<ColorRgb> _palette = new();
    private bool _isInitialized;

    public ColorQuantizer(int colorCount)
    {
        _colorCount = colorCount;
    }

    public void BuildPalette(List<ColorRgb> colors)
    {
        if (colors.Count <= _colorCount)
        {
            _palette = colors.Distinct().ToList();
            _isInitialized = true;
            return;
        }

        // K-means clustering for color quantization
        _palette = KMeansClustering(colors, _colorCount);
        _isInitialized = true;
    }

    public ColorRgb Quantize(ColorRgb color)
    {
        if (!_isInitialized || _palette.Count == 0)
            return color;

        // Find nearest color in palette
        var nearest = _palette[0];
        var minDistance = ColorDistance(color, nearest);

        for (int i = 1; i < _palette.Count; i++)
        {
            var distance = ColorDistance(color, _palette[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                nearest = _palette[i];
            }
        }

        return nearest;
    }

    private List<ColorRgb> KMeansClustering(List<ColorRgb> colors, int k)
    {
        var random = new Random(42);

        // Initialize centroids randomly
        var centroids = colors.OrderBy(_ => random.Next()).Take(k).ToList();
        var previousCentroids = new List<ColorRgb>();

        const int maxIterations = 20;
        int iteration = 0;

        while (iteration < maxIterations && !CentroidsEqual(centroids, previousCentroids))
        {
            previousCentroids = centroids.ToList();

            // Assign colors to nearest centroid
            var clusters = new List<ColorRgb>[k];
            for (int i = 0; i < k; i++)
                clusters[i] = new List<ColorRgb>();

            foreach (var color in colors)
            {
                var nearestIndex = 0;
                var minDistance = ColorDistance(color, centroids[0]);

                for (int i = 1; i < k; i++)
                {
                    var distance = ColorDistance(color, centroids[i]);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        nearestIndex = i;
                    }
                }

                clusters[nearestIndex].Add(color);
            }

            // Update centroids
            for (int i = 0; i < k; i++)
            {
                if (clusters[i].Count > 0)
                {
                    var avgR = (byte)clusters[i].Average(c => c.R);
                    var avgG = (byte)clusters[i].Average(c => c.G);
                    var avgB = (byte)clusters[i].Average(c => c.B);
                    centroids[i] = new ColorRgb(avgR, avgG, avgB);
                }
            }

            iteration++;
        }

        return centroids;
    }

    private double ColorDistance(ColorRgb c1, ColorRgb c2)
    {
        var dr = c1.R - c2.R;
        var dg = c1.G - c2.G;
        var db = c1.B - c2.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }

    private bool CentroidsEqual(List<ColorRgb> c1, List<ColorRgb> c2)
    {
        if (c1.Count != c2.Count)
            return false;

        for (int i = 0; i < c1.Count; i++)
        {
            if (!c1[i].Equals(c2[i]))
                return false;
        }

        return true;
    }
}
