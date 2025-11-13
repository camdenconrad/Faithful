using System.Collections.Generic;
using System.Linq;

namespace NNImage.Models;

/// <summary>
/// Stores learned rules about which colors appear in which structural contexts
/// </summary>
public class StructuralRuleGraph
{
    /// <summary>
    /// Maps structure class -> color -> frequency
    /// </summary>
    public Dictionary<int, Dictionary<uint, int>> StructureColorFrequency { get; }

    /// <summary>
    /// Maps structure class -> normalized color probabilities
    /// </summary>
    public Dictionary<int, Dictionary<uint, double>> StructureColorProbability { get; }

    /// <summary>
    /// Number of structure classes
    /// </summary>
    public int NumStructureClasses { get; set; }

    public StructuralRuleGraph()
    {
        StructureColorFrequency = new Dictionary<int, Dictionary<uint, int>>();
        StructureColorProbability = new Dictionary<int, Dictionary<uint, double>>();
        NumStructureClasses = 0;
    }

    /// <summary>
    /// Records that a color appeared in a specific structure class
    /// </summary>
    public void RecordObservation(int structureClass, uint color)
    {
        if (!StructureColorFrequency.ContainsKey(structureClass))
        {
            StructureColorFrequency[structureClass] = new Dictionary<uint, int>();
        }

        if (!StructureColorFrequency[structureClass].ContainsKey(color))
        {
            StructureColorFrequency[structureClass][color] = 0;
        }

        StructureColorFrequency[structureClass][color]++;
    }

    /// <summary>
    /// Normalizes frequencies to probabilities
    /// </summary>
    public void Normalize()
    {
        StructureColorProbability.Clear();

        foreach (var (structureClass, colorFreq) in StructureColorFrequency)
        {
            int total = colorFreq.Values.Sum();
            if (total == 0) continue;

            StructureColorProbability[structureClass] = new Dictionary<uint, double>();
            foreach (var (color, freq) in colorFreq)
            {
                StructureColorProbability[structureClass][color] = (double)freq / total;
            }
        }
    }

    /// <summary>
    /// Gets color probability for a given structure class
    /// </summary>
    public double GetColorProbability(int structureClass, uint color)
    {
        if (StructureColorProbability.TryGetValue(structureClass, out var colorProbs))
        {
            if (colorProbs.TryGetValue(color, out var prob))
            {
                return prob;
            }
        }
        return 0.0;
    }

    /// <summary>
    /// Gets all colors observed in a structure class
    /// </summary>
    public HashSet<uint> GetColorsForStructure(int structureClass)
    {
        if (StructureColorProbability.TryGetValue(structureClass, out var colorProbs))
        {
            return new HashSet<uint>(colorProbs.Keys);
        }
        return new HashSet<uint>();
    }
}
