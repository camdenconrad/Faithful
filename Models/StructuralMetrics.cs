using System;

namespace NNImage.Models;

/// <summary>
/// Represents structural metrics computed for a pixel/tile
/// </summary>
public class StructuralMetrics
{
    /// <summary>
    /// Luminance value: L = 0.299R + 0.587G + 0.114B
    /// </summary>
    public double Luminance { get; set; }

    /// <summary>
    /// Gradient magnitude from Sobel or Scharr filter
    /// </summary>
    public double GradientMagnitude { get; set; }

    /// <summary>
    /// Local entropy (diversity of neighboring intensities)
    /// </summary>
    public double Entropy { get; set; }

    /// <summary>
    /// Quantized structure class (for grouping similar structures)
    /// </summary>
    public int StructureClass { get; set; }

    public StructuralMetrics(double luminance, double gradientMagnitude, double entropy)
    {
        Luminance = luminance;
        GradientMagnitude = gradientMagnitude;
        Entropy = entropy;
        StructureClass = -1; // Unassigned
    }

    /// <summary>
    /// Computes similarity distance to another structural metric
    /// </summary>
    public double DistanceTo(StructuralMetrics other)
    {
        double lumDiff = Luminance - other.Luminance;
        double gradDiff = GradientMagnitude - other.GradientMagnitude;
        double entropyDiff = Entropy - other.Entropy;

        return Math.Sqrt(lumDiff * lumDiff + gradDiff * gradDiff + entropyDiff * entropyDiff);
    }
}
