using repliKate;

namespace repliKate;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

// ===============================================================================
// QUANTIZATION MODES
// ===============================================================================
public enum QuantizationMode
{
    None,   // Raw 32-bit floats (Tensor) - Maximum precision, 4x memory
    Bit8,   // 8-bit quantization (256 values) - Compact, lossy
    Bit16   // 16-bit quantization (65,536 values) - Balanced precision/memory
}

// ===============================================================================
// OPTIMIZATION: QuantizedTensor - Memory reduction with quantization
// ===============================================================================
public class QuantizedTensor
{
    public int[] Shape { get; private set; }
    public ushort[] QuantizedData { get; private set; }  // CHANGED: ushort (16-bit) instead of byte
    public float Min { get; private set; }
    public float Max { get; private set; }
    public int Size => QuantizedData.Length;

    public QuantizedTensor(float[] data, int[] shape)
    {
        Shape = shape;
        Min = data.Min();
        Max = data.Max();

        QuantizedData = new ushort[data.Length];  // CHANGED: ushort instead of byte
        float range = Max - Min;

        if (range < 1e-8f)
        {
            Array.Fill(QuantizedData, (ushort)32767);  // CHANGED: midpoint of 16-bit
        }
        else
        {
            for (int i = 0; i < data.Length; i++)
            {
                float normalized = (data[i] - Min) / range;
                QuantizedData[i] = (ushort)Math.Clamp((int)(normalized * 65535), 0, 65535);  // CHANGED: 65535 instead of 255
            }
        }
    }

    public QuantizedTensor(int[] shape)
    {
        Shape = shape;
        int size = 1;
        foreach (int dim in shape) size *= dim;
        QuantizedData = new ushort[size];  // CHANGED: ushort instead of byte
        Min = 0;
        Max = 1;
    }

    public float[] Dequantize()
    {
        float[] result = new float[QuantizedData.Length];
        float range = Max - Min;

        for (int i = 0; i < QuantizedData.Length; i++)
        {
            result[i] = Min + (QuantizedData[i] / 65535.0f) * range;  // CHANGED: 65535 instead of 255
        }

        return result;
    }

    public float CosineSimilarityQuantized(QuantizedTensor other)
    {
        if (Size != other.Size) return 0;

        long dot = 0;
        long magA = 0;
        long magB = 0;

        for (int i = 0; i < QuantizedData.Length; i++)
        {
            long a = QuantizedData[i];  // CHANGED: long to handle 16-bit values
            long b = other.QuantizedData[i];

            dot += a * b;
            magA += a * a;
            magB += b * b;
        }

        if (magA == 0 || magB == 0) return 0;
        return (float)(dot / (Math.Sqrt(magA) * Math.Sqrt(magB)));
    }

    public QuantizedTensor Clone()
    {
        var clone = new QuantizedTensor(Shape);
        Array.Copy(QuantizedData, clone.QuantizedData, QuantizedData.Length);
        clone.Min = Min;
        clone.Max = Max;
        return clone;
    }

    public static QuantizedTensor FromTensor(Tensor tensor)
    {
        return new QuantizedTensor(tensor.Data, tensor.Shape);
    }

    public Tensor ToTensor()
    {
        Tensor t = new Tensor(Shape);
        float[] data = Dequantize();
        Array.Copy(data, t.Data, data.Length);
        return t;
    }
}

// ===============================================================================
// NEW: DELTA REGRESSION - State tracking for recurrent-style predictions
// ===============================================================================
public class DeltaState
{
    public Tensor CurrentValue { get; set; }
    public Tensor Velocity { get; set; }  // First-order derivative (delta)
    public Tensor Acceleration { get; set; }  // Second-order derivative
    public Tensor SmoothedValue { get; set; }  // Exponential moving average
    public float Momentum { get; set; }  // Momentum factor for velocity
    public float SmoothingFactor { get; set; }  // EMA smoothing factor (alpha)
    public int StepCount { get; set; }

    public DeltaState(int tensorSize, float momentum = 0.9f, float smoothingFactor = 0.7f)
    {
        CurrentValue = new Tensor(tensorSize);
        Velocity = new Tensor(tensorSize);
        Acceleration = new Tensor(tensorSize);
        SmoothedValue = new Tensor(tensorSize);
        Momentum = momentum;
        SmoothingFactor = smoothingFactor;
        StepCount = 0;
    }

    public DeltaState Clone()
    {
        var clone = new DeltaState(CurrentValue.Size, Momentum, SmoothingFactor);
        Array.Copy(CurrentValue.Data, clone.CurrentValue.Data, CurrentValue.Size);
        Array.Copy(Velocity.Data, clone.Velocity.Data, Velocity.Size);
        Array.Copy(Acceleration.Data, clone.Acceleration.Data, Acceleration.Size);
        Array.Copy(SmoothedValue.Data, clone.SmoothedValue.Data, SmoothedValue.Size);
        clone.StepCount = StepCount;
        return clone;
    }

    /// <summary>
    /// Update state with new observation (like RNN hidden state update)
    /// </summary>
    public void Update(Tensor newValue)
    {
        if (StepCount > 0)
        {
            // Calculate new velocity (first derivative / delta)
            Tensor newVelocity = new Tensor(newValue.Size);
            for (int i = 0; i < newValue.Size; i++)
            {
                newVelocity.Data[i] = newValue.Data[i] - CurrentValue.Data[i];
            }

            // Calculate acceleration (second derivative) with momentum
            if (StepCount > 1)
            {
                for (int i = 0; i < newValue.Size; i++)
                {
                    float velocityChange = newVelocity.Data[i] - Velocity.Data[i];
                    Acceleration.Data[i] = Momentum * Acceleration.Data[i] +
                                          (1 - Momentum) * velocityChange;
                }
            }

            // Update velocity with momentum (similar to LSTM forget gate)
            for (int i = 0; i < newValue.Size; i++)
            {
                Velocity.Data[i] = Momentum * Velocity.Data[i] + (1 - Momentum) * newVelocity.Data[i];
            }

            // Update exponential moving average (smoothed value)
            for (int i = 0; i < newValue.Size; i++)
            {
                SmoothedValue.Data[i] = SmoothingFactor * newValue.Data[i] +
                                       (1 - SmoothingFactor) * SmoothedValue.Data[i];
            }
        }
        else
        {
            // Initialize smoothed value on first observation
            Array.Copy(newValue.Data, SmoothedValue.Data, newValue.Size);
        }

        Array.Copy(newValue.Data, CurrentValue.Data, newValue.Size);
        StepCount++;
    }

    /// <summary>
    /// Predict next value using physics-based extrapolation
    /// Similar to RNN forward pass but with explicit dynamics
    /// </summary>
    public Tensor PredictNext(int stepsAhead = 1, bool useAcceleration = true, float dampingFactor = 1.0f)
    {
        Tensor prediction = new Tensor(CurrentValue.Size);

        for (int i = 0; i < CurrentValue.Size; i++)
        {
            float value = CurrentValue.Data[i];
            float vel = Velocity.Data[i] * dampingFactor;  // Apply damping
            float acc = useAcceleration ? Acceleration.Data[i] * dampingFactor : 0f;

            // Physics-based extrapolation: x(t+dt) = x(t) + v*dt + 0.5*a*dt²
            // This is similar to how RNNs extrapolate hidden states
            prediction.Data[i] = value + vel * stepsAhead + 0.5f * acc * stepsAhead * stepsAhead;
        }

        return prediction;
    }

    /// <summary>
    /// Predict using smoothed value (more stable, less reactive)
    /// </summary>
    public Tensor PredictNextSmoothed(int stepsAhead = 1, float dampingFactor = 0.95f)
    {
        Tensor prediction = new Tensor(SmoothedValue.Size);

        for (int i = 0; i < SmoothedValue.Size; i++)
        {
            float vel = Velocity.Data[i] * dampingFactor;
            prediction.Data[i] = SmoothedValue.Data[i] + vel * stepsAhead;
        }

        return prediction;
    }

    /// <summary>
    /// Get confidence in predictions based on velocity/acceleration stability
    /// High stability = high confidence
    /// </summary>
    public float GetPredictionConfidence()
    {
        if (StepCount < 3) return 0.5f;

        // Calculate velocity and acceleration variance
        float velMagnitude = 0;
        float accMagnitude = 0;

        for (int i = 0; i < Velocity.Size; i++)
        {
            velMagnitude += Math.Abs(Velocity.Data[i]);
            accMagnitude += Math.Abs(Acceleration.Data[i]);
        }

        velMagnitude /= Velocity.Size;
        accMagnitude /= Acceleration.Size;

        // Lower magnitude = more stable = higher confidence
        float velConfidence = 1.0f / (1.0f + velMagnitude);
        float accConfidence = 1.0f / (1.0f + accMagnitude);

        return 0.6f * velConfidence + 0.4f * accConfidence;
    }
}

// ===============================================================================
// OPTIMIZATION: TensorNode now stores indices instead of cloning tensors
// ===============================================================================
public class TensorNode
{
    public int SequenceIndex { get; set; }  // FIXED: Changed to setter for index updates
    private Dictionary<int, float> nextIndices;
    private float totalWeight;

    public TensorNode(int sequenceIndex)
    {
        SequenceIndex = sequenceIndex;
        nextIndices = new Dictionary<int, float>();
        totalWeight = 0;
    }

    public void RecordNext(int nextIndex, float weight = 1.0f)
    {
        if (!nextIndices.ContainsKey(nextIndex))
            nextIndices[nextIndex] = 0;

        nextIndices[nextIndex] += weight;
        totalWeight += weight;
    }

    // FIXED: Add method to update indices after removal
    public void UpdateIndicesAfterRemoval(int removedCount)
    {
        SequenceIndex = Math.Max(0, SequenceIndex - removedCount);

        var updatedNextIndices = new Dictionary<int, float>();
        foreach (var kvp in nextIndices)
        {
            int newIndex = kvp.Key - removedCount;
            if (newIndex >= 0)
            {
                updatedNextIndices[newIndex] = kvp.Value;
            }
            else
            {
                // This transition points to removed data, subtract its weight
                totalWeight -= kvp.Value;
            }
        }
        nextIndices = updatedNextIndices;
    }

    // FIXED: Add method to check if node is still valid
    public bool IsValid(int maxSequenceLength)
    {
        return SequenceIndex >= 0 && SequenceIndex < maxSequenceLength && nextIndices.Count > 0;
    }

    public int GetMostLikelyNextIndex()
    {
        if (nextIndices.Count == 0) return -1;

        int bestIndex = -1;
        float bestWeight = 0;

        foreach (var kvp in nextIndices)
        {
            if (kvp.Value > bestWeight)
            {
                bestWeight = kvp.Value;
                bestIndex = kvp.Key;
            }
        }

        return bestIndex;
    }

    public List<(int index, float probability)> GetNextProbabilities()
    {
        if (totalWeight == 0) return new List<(int, float)>();

        var result = new List<(int, float)>();
        foreach (var kvp in nextIndices)
        {
            result.Add((kvp.Key, kvp.Value / totalWeight));
        }

        return result.OrderByDescending(x => x.Item2).ToList();
    }

    public List<(int index, float score)> GetTopNext(int count = 5)
    {
        if (nextIndices.Count == 0) return new List<(int, float)>();
        if (totalWeight == 0) return new List<(int, float)>();

        var scored = new List<(int, float)>();
        foreach (var kvp in nextIndices)
        {
            scored.Add((kvp.Key, kvp.Value / totalWeight));
        }

        return scored
            .OrderByDescending(x => x.Item2)
            .Take(count)
            .ToList();
    }

    public Dictionary<int, float> GetNextIndices() => new Dictionary<int, float>(nextIndices);
    public float GetTotalWeight() => totalWeight;
}

public class TensorSequenceTree
{
    private List<TensorNode> nodes;
    private List<QuantizedTensor> fullSequence;
    private Dictionary<int, List<(List<int> contextIndices, Dictionary<int, float> nextIndices)>> nGrams;
    private Dictionary<int, Dictionary<int, int>> nGramHashIndex;
    private Dictionary<int, int> tensorHashIndex;
    private int maxContextWindow;
    private int tensorSize;
    private float similarityThreshold;
    private float baseSimilarityThreshold;
    private float temporalDecayFactor;
    private long transitionCounter;
    private float temperature;
    private float explorationRate;
    private Random random;
    private List<(Tensor[] sequence, float outcome, long timestamp)> experienceBuffer;
    private int experienceBufferCapacity;
    private bool useExperienceReplay;
    private bool useQuantization;

    // NEW: Delta regression state tracking
    private DeltaState deltaState;
    private bool useDeltaRegression;
    private List<DeltaState> deltaHistory;  // History of delta states for advanced predictions
    private int maxDeltaHistorySize;

    private const int MAX_SEQUENCE_LENGTH = 5000000;
    private const int MAX_NODES = 100000000;
    private const int MAX_NGRAM_ENTRIES_PER_N = 5000;
    private const int DEFAULT_EXPERIENCE_CAPACITY = 10000000;
    private const int DEFAULT_DELTA_HISTORY_SIZE = 50;

    public TensorSequenceTree(int maxContextWindow = 100,
        float similarityThreshold = 0.95f,
        float temporalDecay = 0.9995f,
        float temperature = 1.0f,
        float explorationRate = 0.05f,
        bool enableExperienceReplay = true,
        int experienceCapacity = DEFAULT_EXPERIENCE_CAPACITY,
        QuantizationMode quantizationMode = QuantizationMode.None,
        bool useQuantization = true,
        bool enableDeltaRegression = true,
        int deltaHistorySize = DEFAULT_DELTA_HISTORY_SIZE)
    {
        nodes = new List<TensorNode>();
        fullSequence = new List<QuantizedTensor>();
        nGrams = new Dictionary<int, List<(List<int>, Dictionary<int, float>)>>();
        nGramHashIndex = new Dictionary<int, Dictionary<int, int>>();
        tensorHashIndex = new Dictionary<int, int>();
        random = new Random();
        experienceBuffer = new List<(Tensor[], float, long)>();
        experienceBufferCapacity = experienceCapacity;
        useExperienceReplay = enableExperienceReplay;
        this.useQuantization = useQuantization;

        // NEW: Initialize delta regression components
        this.useDeltaRegression = enableDeltaRegression;
        this.deltaHistory = new List<DeltaState>();
        this.maxDeltaHistorySize = deltaHistorySize;

        this.maxContextWindow = Math.Max(2, Math.Min(maxContextWindow, 100));
        this.similarityThreshold = similarityThreshold;
        this.baseSimilarityThreshold = similarityThreshold;
        this.temporalDecayFactor = Math.Clamp(temporalDecay, 0.95f, 0.9999f);
        this.temperature = Math.Max(0.1f, temperature);
        this.explorationRate = Math.Clamp(explorationRate, 0f, 1f);
        this.transitionCounter = 0;
        tensorSize = 0;

        for (int n = 2; n <= this.maxContextWindow; n++)
        {
            nGrams[n] = new List<(List<int>, Dictionary<int, float>)>();
            nGramHashIndex[n] = new Dictionary<int, int>();
        }
    }

    public int GetMaxContextWindow() => maxContextWindow;
    public int GetTensorSize() => tensorSize;

    /// <summary>
    /// Find where a sequence best fits in the existing learned data
    /// Returns (bestMatchIndex, matchLength, isCompletelyNew)
    /// </summary>
    private (int matchIndex, int matchLength, bool isNew) FindSequenceMatch(Tensor[] sequence)
    {
        if (fullSequence.Count == 0 || sequence.Length == 0)
            return (-1, 0, true);

        int bestMatchIndex = -1;
        int bestMatchLength = 0;
        float bestMatchScore = 0;

        // Convert sequence to quantized for comparison
        var qSequence = sequence.Select(t => QuantizedTensor.FromTensor(t)).ToArray();

        // Search for the best matching position
        for (int i = 0; i <= fullSequence.Count - sequence.Length; i++)
        {
            int matchLength = 0;
            float totalSimilarity = 0;

            // Check how many consecutive tensors match
            for (int j = 0; j < sequence.Length && (i + j) < fullSequence.Count; j++)
            {
                float similarity = qSequence[j].CosineSimilarityQuantized(fullSequence[i + j]);

                if (similarity >= 0.95f) // INCREASED from 0.85 - stricter matching
                {
                    matchLength++;
                    totalSimilarity += similarity;
                }
                else
                {
                    break; // Stop at first mismatch
                }
            }

            if (matchLength > 0)
            {
                float avgSimilarity = totalSimilarity / matchLength;
                float score = matchLength * avgSimilarity;

                if (score > bestMatchScore)
                {
                    bestMatchScore = score;
                    bestMatchIndex = i;
                    bestMatchLength = matchLength;
                }
            }
        }

        // Also check if sequence is a continuation (matches the end)
        int tailCheckSize = Math.Min(sequence.Length, fullSequence.Count);
        int tailMatchLength = 0;
        float tailTotalSimilarity = 0;

        for (int i = 0; i < tailCheckSize; i++)
        {
            int fullSeqIdx = fullSequence.Count - tailCheckSize + i;
            float similarity = qSequence[i].CosineSimilarityQuantized(fullSequence[fullSeqIdx]);

            if (similarity >= 0.95f)  // INCREASED from 0.85 - stricter matching
            {
                tailMatchLength++;
                tailTotalSimilarity += similarity;
            }
            else
            {
                break;
            }
        }

        if (tailMatchLength > 0)
        {
            float tailAvgSimilarity = tailTotalSimilarity / tailMatchLength;
            float tailScore = tailMatchLength * tailAvgSimilarity;

            if (tailScore > bestMatchScore)
            {
                bestMatchIndex = fullSequence.Count - tailMatchLength;
                bestMatchLength = tailMatchLength;
                bestMatchScore = tailScore;
            }
        }

        // Consider it "new" if match is less than 80% of sequence length
        // INCREASED from 30% to 80% to be more conservative
        bool isCompletelyNew = bestMatchLength < (sequence.Length * 0.80f);

        return (bestMatchIndex, bestMatchLength, isCompletelyNew);
    }

    /// <summary>
    /// Find or reuse an existing tensor index (cross-tree pollination)
    /// Returns the index of a matching tensor if found, otherwise -1
    /// </summary>
    private int FindOrReuseTensorIndex(QuantizedTensor tensor, float reuseThreshold = 0.95f)
    {
        // Check hash index first
        int hash = ComputeTensorHash(tensor);
        if (tensorHashIndex.TryGetValue(hash, out int idx))
        {
            if (idx >= 0 && idx < fullSequence.Count)
            {
                float sim = tensor.CosineSimilarityQuantized(fullSequence[idx]);
                if (sim >= reuseThreshold)
                    return idx;
            }
        }

        // Search for highly similar tensors (for cross-tree pollination)
        // Check recent entries for efficiency
        for (int i = fullSequence.Count - 1; i >= Math.Max(0, fullSequence.Count - 1000); i--)
        {
            float sim = tensor.CosineSimilarityQuantized(fullSequence[i]);
            if (sim >= reuseThreshold)
                return i;
        }

        return -1;
    }

    public void LearnWithOutcome(Tensor[] sequence, float outcome)
    {
        if (sequence == null || sequence.Length == 0) return;

        // NEW: Update delta state with sequence
        if (useDeltaRegression)
        {
            UpdateDeltaStateWithSequence(sequence);
        }

        if (useExperienceReplay)
        {
            experienceBuffer.Add((sequence.Select(t => t.Clone()).ToArray(), outcome, transitionCounter));

            if (experienceBuffer.Count > experienceBufferCapacity)
            {
                experienceBuffer = experienceBuffer
                    .OrderByDescending(exp => Math.Abs(exp.outcome))
                    .ThenByDescending(exp => exp.timestamp)
                    .Take(experienceBufferCapacity)
                    .ToList();
            }
        }

        LearnInternal(sequence, outcome);
    }

    /// <summary>
    /// NEW: Update delta state with observed sequence (like RNN state update)
    /// </summary>
    private void UpdateDeltaStateWithSequence(Tensor[] sequence)
    {
        if (sequence == null || sequence.Length == 0) return;

        // Initialize delta state if needed
        if (deltaState == null && sequence.Length > 0)
        {
            deltaState = new DeltaState(sequence[0].Size);
        }

        // Update delta state with each tensor in sequence
        foreach (var tensor in sequence)
        {
            deltaState.Update(tensor);

            // Store snapshot in history for pattern analysis
            if (deltaHistory.Count >= maxDeltaHistorySize)
            {
                deltaHistory.RemoveAt(0);
            }
            deltaHistory.Add(deltaState.Clone());
        }
    }

    private void LearnInternal(Tensor[] sequence, float outcomeWeight)
    {
        if (sequence == null || sequence.Length == 0) return;

        if (tensorSize == 0 && sequence.Length > 0)
        {
            tensorSize = sequence[0].Size;
        }

        foreach (var tensor in sequence)
        {
            if (tensor.Size != tensorSize)
                throw new ArgumentException($"All tensors must have size {tensorSize}");
        }

        AdaptSimilarityThreshold();

        float[] positionWeights = new float[sequence.Length];
        for (int i = 0; i < sequence.Length; i++)
        {
            float progressWeight = 1.0f + (i / (float)sequence.Length) * 2.0f;
            positionWeights[i] = progressWeight * Math.Abs(outcomeWeight);
        }

        // NEW: Find where this sequence fits in existing data
        var (matchIndex, matchLength, isCompletelyNew) = FindSequenceMatch(sequence);

        List<int> sequenceIndices = new List<int>();
        int baseIndex = 0;

        if (isCompletelyNew)
        {
            // Completely new sequence: add to the end (will be managed by rolling window)
            baseIndex = fullSequence.Count;

            foreach (var tensor in sequence)
            {
                var quantized = QuantizedTensor.FromTensor(tensor);

                // NEW: Try to reuse existing tensor for cross-tree pollination
                int existingIndex = FindOrReuseTensorIndex(quantized, similarityThreshold);

                if (existingIndex >= 0)
                {
                    // Reuse existing tensor (creates graph connections)
                    sequenceIndices.Add(existingIndex);
                }
                else
                {
                    // Add new tensor
                    fullSequence.Add(quantized);
                    int hash = ComputeTensorHash(quantized);
                    tensorHashIndex[hash] = fullSequence.Count - 1;
                    sequenceIndices.Add(fullSequence.Count - 1);
                }
            }
        }
        else
        {
            // Sequence matches existing data: insert where it fits
            baseIndex = matchIndex;

            // For matching portion, reuse existing indices
            for (int i = 0; i < matchLength; i++)
            {
                sequenceIndices.Add(matchIndex + i);
            }

            // For non-matching portion, add new tensors after the match
            for (int i = matchLength; i < sequence.Length; i++)
            {
                var quantized = QuantizedTensor.FromTensor(sequence[i]);

                // NEW: Try to reuse existing tensor for cross-tree pollination
                int existingIndex = FindOrReuseTensorIndex(quantized, similarityThreshold);

                if (existingIndex >= 0)
                {
                    sequenceIndices.Add(existingIndex);
                }
                else
                {
                    // Insert after the matching portion
                    int insertPos = matchIndex + matchLength + (i - matchLength);

                    if (insertPos < fullSequence.Count)
                    {
                        // Insert in the middle
                        fullSequence.Insert(insertPos, quantized);

                        // Update all indices that come after insertion point
                        UpdateAllIndicesAfterInsertion(insertPos, 1);

                        sequenceIndices.Add(insertPos);
                    }
                    else
                    {
                        // Append to end
                        fullSequence.Add(quantized);
                        sequenceIndices.Add(fullSequence.Count - 1);
                    }

                    int hash = ComputeTensorHash(quantized);
                    tensorHashIndex[hash] = sequenceIndices[sequenceIndices.Count - 1];
                }
            }
        }

        // Handle rolling window with index updates
        int toRemove = 0;
        if (fullSequence.Count > MAX_SEQUENCE_LENGTH)
        {
            toRemove = fullSequence.Count - MAX_SEQUENCE_LENGTH;
            fullSequence.RemoveRange(0, toRemove);

            UpdateAllIndicesAfterRemoval(toRemove);
            UpdateNGramIndicesAfterRemoval(toRemove);
            RebuildTensorHashIndex();

            baseIndex = Math.Max(0, baseIndex - toRemove);
            for (int i = 0; i < sequenceIndices.Count; i++)
            {
                sequenceIndices[i] = Math.Max(0, sequenceIndices[i] - toRemove);
            }
        }

        float temporalWeight = GetTemporalWeight();

        // Record transitions (creates graph structure with cross-connections)
        for (int i = 0; i < sequenceIndices.Count - 1; i++)
        {
            int currentIdx = sequenceIndices[i];
            int nextIdx = sequenceIndices[i + 1];

            float combinedWeight = temporalWeight * positionWeights[i];

            int nodeIndex = FindOrCreateNode(currentIdx);
            if (nodeIndex >= 0)
            {
                // This naturally creates graph connections when multiple paths lead to same node
                nodes[nodeIndex].RecordNext(nextIdx, combinedWeight);
            }
            transitionCounter++;
        }

        FindOrCreateNode(sequenceIndices[sequenceIndices.Count - 1]);
        BuildNGramsFromIndices(sequenceIndices, baseIndex, temporalWeight * Math.Abs(outcomeWeight));
        PruneIfNeeded();
    }

    // NEW: Method to update all indices after insertion
    private void UpdateAllIndicesAfterInsertion(int insertPos, int insertCount)
    {
        // Update all nodes
        foreach (var node in nodes)
        {
            if (node.SequenceIndex >= insertPos)
            {
                node.SequenceIndex += insertCount;
            }

            // Update next indices
            var nextIndices = node.GetNextIndices();
            var updatedNextIndices = new Dictionary<int, float>();

            foreach (var kvp in nextIndices)
            {
                int newIndex = kvp.Key >= insertPos ? kvp.Key + insertCount : kvp.Key;
                updatedNextIndices[newIndex] = kvp.Value;
            }

            // Clear and rebuild next indices
            foreach (var kvp in updatedNextIndices)
            {
                node.RecordNext(kvp.Key, kvp.Value);
            }
        }

        // Update n-grams
        foreach (var n in nGrams.Keys.ToList())
        {
            var updatedEntries = new List<(List<int> contextIndices, Dictionary<int, float> nextIndices)>();

            foreach (var entry in nGrams[n])
            {
                var updatedContextIndices = entry.contextIndices.Select(idx => idx >= insertPos ? idx + insertCount : idx).ToList();

                var updatedNextIndices = new Dictionary<int, float>();
                foreach (var kvp in entry.nextIndices)
                {
                    int newNextIndex = kvp.Key >= insertPos ? kvp.Key + insertCount : kvp.Key;
                    updatedNextIndices[newNextIndex] = kvp.Value;
                }

                updatedEntries.Add((updatedContextIndices, updatedNextIndices));
            }

            nGrams[n] = updatedEntries;

            // Rebuild hash index
            nGramHashIndex[n].Clear();
            for (int i = 0; i < updatedEntries.Count; i++)
            {
                int hash = ComputeContextHashFromIndices(updatedEntries[i].contextIndices);
                nGramHashIndex[n][hash] = i;
            }
        }
    }

    // FIXED: New method to update all node indices after removal
    private void UpdateAllIndicesAfterRemoval(int removedCount)
    {
        // Update all nodes
        for (int i = nodes.Count - 1; i >= 0; i--)
        {
            nodes[i].UpdateIndicesAfterRemoval(removedCount);

            // Remove nodes that are no longer valid
            if (!nodes[i].IsValid(fullSequence.Count))
            {
                nodes.RemoveAt(i);
            }
        }
    }

    // FIXED: New method to update all n-gram indices after removal
    private void UpdateNGramIndicesAfterRemoval(int removedCount)
    {
        foreach (var n in nGrams.Keys.ToList())
        {
            var updatedEntries = new List<(List<int> contextIndices, Dictionary<int, float> nextIndices)>();

            foreach (var entry in nGrams[n])
            {
                // Update context indices
                var updatedContextIndices = entry.contextIndices.Select(idx => idx - removedCount).ToList();

                // Check if all context indices are still valid
                bool contextValid = updatedContextIndices.All(idx => idx >= 0 && idx < fullSequence.Count);

                if (!contextValid)
                    continue;

                // Update next indices
                var updatedNextIndices = new Dictionary<int, float>();
                foreach (var kvp in entry.nextIndices)
                {
                    int newNextIndex = kvp.Key - removedCount;
                    if (newNextIndex >= 0 && newNextIndex < fullSequence.Count)
                    {
                        updatedNextIndices[newNextIndex] = kvp.Value;
                    }
                }

                // Only keep entries with valid next indices
                if (updatedNextIndices.Count > 0)
                {
                    updatedEntries.Add((updatedContextIndices, updatedNextIndices));
                }
            }

            nGrams[n] = updatedEntries;

            // Rebuild hash index for this n
            nGramHashIndex[n].Clear();
            for (int i = 0; i < updatedEntries.Count; i++)
            {
                int hash = ComputeContextHashFromIndices(updatedEntries[i].contextIndices);
                nGramHashIndex[n][hash] = i;
            }
        }
    }

    public void Learn(Tensor[] sequence)
    {
        LearnWithOutcome(sequence, 1.0f);
    }

    private int FindOrCreateNode(int sequenceIndex)
    {
        if (sequenceIndex < 0 || sequenceIndex >= fullSequence.Count)
            return -1;

        if (nodes.Count >= MAX_NODES)
        {
            int bestIdx = -1;
            float bestSim = 0;
            for (int i = 0; i < nodes.Count; i++)
            {
                int nodeSeqIdx = nodes[i].SequenceIndex;
                if (nodeSeqIdx >= 0 && nodeSeqIdx < fullSequence.Count)
                {
                    float sim = fullSequence[sequenceIndex].CosineSimilarityQuantized(fullSequence[nodeSeqIdx]);
                    if (sim > bestSim)
                    {
                        bestSim = sim;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx >= 0 && bestSim >= similarityThreshold)
            {
                return bestIdx;
            }
            return -1;
        }

        for (int i = 0; i < nodes.Count; i++)
        {
            int nodeSeqIdx = nodes[i].SequenceIndex;
            if (nodeSeqIdx >= 0 && nodeSeqIdx < fullSequence.Count)
            {
                if (fullSequence[sequenceIndex].CosineSimilarityQuantized(fullSequence[nodeSeqIdx]) >= similarityThreshold)
                {
                    return i;
                }
            }
        }

        nodes.Add(new TensorNode(sequenceIndex));
        return nodes.Count - 1;
    }

    private void BuildNGramsFromIndices(List<int> sequenceIndices, int baseIndex, float temporalWeight)
    {
        for (int n = 2; n <= maxContextWindow; n++)
        {
            if (sequenceIndices.Count < n)
                continue;

            for (int i = 0; i <= sequenceIndices.Count - n; i++)
            {
                List<int> contextIndices = sequenceIndices.GetRange(i, n - 1);
                int nextIndex = sequenceIndices[i + n - 1];
                AddNGramOptimized(n, contextIndices, nextIndex, temporalWeight);
            }
        }
    }

    private void AddNGramOptimized(int n, List<int> contextIndices, int nextIndex, float weight)
    {
        int contextHash = ComputeContextHashFromIndices(contextIndices);

        if (nGramHashIndex[n].TryGetValue(contextHash, out int existingIndex))
        {
            var entry = nGrams[n][existingIndex];
            if (ContextIndicesMatch(entry.contextIndices, contextIndices))
            {
                if (!entry.nextIndices.ContainsKey(nextIndex))
                    entry.nextIndices[nextIndex] = 0;
                entry.nextIndices[nextIndex] += weight;
                return;
            }
        }

        var newEntry = (new List<int>(contextIndices), new Dictionary<int, float> { { nextIndex, weight } });
        nGrams[n].Add(newEntry);
        nGramHashIndex[n][contextHash] = nGrams[n].Count - 1;
    }

    private bool ContextIndicesMatch(List<int> ctx1, List<int> ctx2)
    {
        if (ctx1.Count != ctx2.Count) return false;

        for (int i = 0; i < ctx1.Count; i++)
        {
            if (ctx1[i] < 0 || ctx1[i] >= fullSequence.Count || ctx2[i] < 0 || ctx2[i] >= fullSequence.Count)
                return false;

            float sim = fullSequence[ctx1[i]].CosineSimilarityQuantized(fullSequence[ctx2[i]]);
            if (sim < similarityThreshold)
                return false;
        }

        return true;
    }

    public Tensor[] PredictNext(Tensor[] context, int count = 1, bool useBlending = false, bool useStochastic = false)
    {
        if (context == null || context.Length == 0)
            return new Tensor[0];

        List<Tensor> predictions = new List<Tensor>();
        List<int> currentContextIndices = new List<int>();

        foreach (var tensor in context)
        {
            int idx = FindSimilarTensorIndex(tensor);
            if (idx >= 0)
                currentContextIndices.Add(idx);
        }

        for (int i = 0; i < count; i++)
        {
            int nextIdx;

            if (useStochastic)
            {
                nextIdx = SampleNextStochasticIndex(currentContextIndices, boostRare: true);
            }
            else
            {
                nextIdx = PredictSingleNextIndex(currentContextIndices, useBlending);
            }

            if (nextIdx < 0 || nextIdx >= fullSequence.Count)
                break;

            Tensor next = fullSequence[nextIdx].ToTensor();
            predictions.Add(next);

            currentContextIndices.Add(nextIdx);
            if (currentContextIndices.Count > maxContextWindow)
                currentContextIndices.RemoveAt(0);
        }

        return predictions.ToArray();
    }

    // ===============================================================================
    // NEW: DELTA REGRESSION PREDICTION METHODS
    // ===============================================================================

    /// <summary>
    /// Predict next tensor using delta (velocity/acceleration) extrapolation
    /// Similar to how RNNs propagate hidden states forward
    /// </summary>
    public Tensor PredictNextDelta(Tensor[] context, int stepsAhead = 1, bool useAcceleration = true, float dampingFactor = 0.95f)
    {
        if (!useDeltaRegression || deltaState == null || context == null || context.Length == 0)
            return null;

        // Update delta state with current context
        foreach (var tensor in context)
        {
            deltaState.Update(tensor);
        }

        return deltaState.PredictNext(stepsAhead, useAcceleration, dampingFactor);
    }

    /// <summary>
    /// Predict using smoothed exponential moving average (more stable)
    /// </summary>
    public Tensor PredictNextDeltaSmoothed(Tensor[] context, int stepsAhead = 1, float dampingFactor = 0.95f)
    {
        if (!useDeltaRegression || deltaState == null || context == null || context.Length == 0)
            return null;

        foreach (var tensor in context)
        {
            deltaState.Update(tensor);
        }

        return deltaState.PredictNextSmoothed(stepsAhead, dampingFactor);
    }

    /// <summary>
    /// Predict multiple steps ahead using delta regression (recurrent extrapolation)
    /// </summary>
    public Tensor[] PredictMultipleDelta(Tensor[] context, int count = 1, bool useAcceleration = true, float dampingFactor = 0.95f)
    {
        if (!useDeltaRegression || count <= 0 || context == null || context.Length == 0)
            return new Tensor[0];

        List<Tensor> predictions = new List<Tensor>();

        // Initialize with context
        foreach (var tensor in context)
        {
            deltaState.Update(tensor);
        }

        // Generate predictions
        for (int i = 0; i < count; i++)
        {
            Tensor prediction = deltaState.PredictNext(1, useAcceleration, dampingFactor);
            if (prediction == null) break;

            predictions.Add(prediction);

            // Update state with prediction for next iteration (autoregressive)
            deltaState.Update(prediction);
        }

        return predictions.ToArray();
    }

    /// <summary>
    /// Hybrid prediction: blend traditional retrieval with delta extrapolation
    /// Similar to ensemble methods in ML
    /// </summary>
    public Tensor PredictNextHybridDelta(Tensor[] context, float deltaWeight = 0.4f, bool useAcceleration = true)
    {
        if (context == null || context.Length == 0) return null;

        // Get retrieval-based prediction
        var retrievalPredictions = PredictNext(context, 1, useBlending: false, useStochastic: false);
        Tensor retrievalPred = (retrievalPredictions != null && retrievalPredictions.Length > 0)
            ? retrievalPredictions[0]
            : null;

        // Get delta-based prediction
        Tensor deltaPred = useDeltaRegression
            ? PredictNextDelta(context, 1, useAcceleration)
            : null;

        // Blend predictions
        if (retrievalPred != null && deltaPred != null)
        {
            Tensor blended = new Tensor(retrievalPred.Size);
            float retrievalWeight = 1.0f - deltaWeight;

            for (int i = 0; i < retrievalPred.Size; i++)
            {
                blended.Data[i] = retrievalPred.Data[i] * retrievalWeight +
                                 deltaPred.Data[i] * deltaWeight;
            }

            return blended;
        }
        else if (retrievalPred != null)
        {
            return retrievalPred;
        }
        else if (deltaPred != null)
        {
            return deltaPred;
        }

        return null;
    }

    /// <summary>
    /// Ensemble prediction: generate multiple predictions using different methods
    /// Returns weighted average based on prediction confidence
    /// </summary>
    public Tensor PredictNextEnsemble(Tensor[] context, bool includeRegression = true, bool includeDelta = true)
    {
        if (context == null || context.Length == 0) return null;

        var predictions = new List<(Tensor tensor, float weight)>();

        // Retrieval-based prediction (weight based on match quality)
        var retrievalPreds = PredictNext(context, 1, useBlending: false, useStochastic: false);
        if (retrievalPreds != null && retrievalPreds.Length > 0)
        {
            predictions.Add((retrievalPreds[0], 0.35f));
        }

        // Regression-based prediction
        if (includeRegression)
        {
            var regressionPred = PredictNextRegression(context, noveltyBias: 0.1f);
            if (regressionPred != null)
            {
                predictions.Add((regressionPred, 0.30f));
            }
        }

        // Delta-based prediction (weight by confidence)
        if (includeDelta && useDeltaRegression && deltaState != null)
        {
            var deltaPred = PredictNextDelta(context, 1, useAcceleration: true);
            if (deltaPred != null)
            {
                float confidence = deltaState.GetPredictionConfidence();
                predictions.Add((deltaPred, 0.25f * confidence));
            }
        }

        // Trend-based regression
        if (context.Length >= 2)
        {
            var trendPred = PredictNextRegressionWithTrend(context, trendWeight: 0.5f, noveltyBias: 0.05f);
            if (trendPred != null)
            {
                predictions.Add((trendPred, 0.10f));
            }
        }

        if (predictions.Count == 0) return null;

        // Normalize weights
        float totalWeight = predictions.Sum(p => p.weight);
        if (totalWeight == 0) return null;

        // Create weighted ensemble
        Tensor ensemble = new Tensor(tensorSize);
        Array.Fill(ensemble.Data, 0f);

        foreach (var (tensor, weight) in predictions)
        {
            float normalizedWeight = weight / totalWeight;
            for (int i = 0; i < tensorSize; i++)
            {
                ensemble.Data[i] += tensor.Data[i] * normalizedWeight;
            }
        }

        return ensemble;
    }

    /// <summary>
    /// Get prediction confidence based on delta state stability
    /// Returns 0-1 confidence score
    /// </summary>
    public float GetDeltaPredictionConfidence()
    {
        if (!useDeltaRegression || deltaState == null) return 0.5f;
        return deltaState.GetPredictionConfidence();
    }

    /// <summary>
    /// Analyze velocity patterns in delta history
    /// Returns average velocity magnitude and trend direction
    /// </summary>
    public (float avgVelocity, float trendDirection) AnalyzeDeltaTrends()
    {
        if (!useDeltaRegression || deltaHistory.Count < 2)
            return (0f, 0f);

        float totalVelocity = 0f;
        float trendSum = 0f;

        for (int i = 0; i < deltaHistory.Count; i++)
        {
            var state = deltaHistory[i];
            float velMag = 0f;

            for (int j = 0; j < state.Velocity.Size; j++)
            {
                velMag += Math.Abs(state.Velocity.Data[j]);
            }

            totalVelocity += velMag / state.Velocity.Size;

            // Trend: positive if velocities are increasing, negative if decreasing
            if (i > 0)
            {
                float prevVelMag = 0f;
                for (int j = 0; j < deltaHistory[i-1].Velocity.Size; j++)
                {
                    prevVelMag += Math.Abs(deltaHistory[i-1].Velocity.Data[j]);
                }
                prevVelMag /= deltaHistory[i-1].Velocity.Size;

                trendSum += (velMag - prevVelMag);
            }
        }

        float avgVelocity = totalVelocity / deltaHistory.Count;
        float trendDirection = deltaHistory.Count > 1 ? trendSum / (deltaHistory.Count - 1) : 0f;

        return (avgVelocity, trendDirection);
    }

    /// <summary>
    /// Get top N predictions with confidence scores
    /// Useful for understanding prediction uncertainty
    /// </summary>
    public List<(Tensor tensor, float confidence)> GetTopPredictions(Tensor[] context, int topN = 5)
    {
        if (context == null || context.Length == 0)
            return new List<(Tensor, float)>();

        var candidates = new Dictionary<int, float>();

        // Convert context to indices
        List<int> contextIndices = new List<int>();
        foreach (var tensor in context)
        {
            int idx = FindSimilarTensorIndex(tensor);
            if (idx >= 0)
                contextIndices.Add(idx);
        }

        // Gather predictions from n-grams
        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && contextIndices.Count >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictionsFromIndices(candidates, contextIndices, n, weight);
            }
        }

        // Add predictions from node graph
        if (contextIndices.Count > 0)
        {
            int lastIdx = contextIndices[contextIndices.Count - 1];
            int nodeIndex = FindNodeBySequenceIndex(lastIdx);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(topN);
                foreach (var (index, score) in topNext)
                {
                    if (!candidates.ContainsKey(index))
                        candidates[index] = 0;
                    candidates[index] += score * 1.0f;
                }
            }
        }

        if (candidates.Count == 0)
            return new List<(Tensor, float)>();

        // Normalize to probabilities
        float total = candidates.Values.Sum();
        if (total == 0)
            return new List<(Tensor, float)>();

        return candidates
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .Where(kvp => kvp.Key >= 0 && kvp.Key < fullSequence.Count)
            .Select(kvp => (fullSequence[kvp.Key].ToTensor(), kvp.Value / total))
            .ToList();
    }

    /// <summary>
    /// Sample next prediction stochastically (useful for diverse predictions)
    /// </summary>
    public Tensor SampleNextStochastic(Tensor[] context, bool boostRare = true)
    {
        if (context == null || context.Length == 0)
            return null;

        List<int> contextIndices = new List<int>();
        foreach (var tensor in context)
        {
            int idx = FindSimilarTensorIndex(tensor);
            if (idx >= 0)
                contextIndices.Add(idx);
        }

        if (random.NextDouble() < explorationRate && fullSequence.Count > 0)
        {
            return fullSequence[random.Next(fullSequence.Count)].ToTensor();
        }

        var candidates = new Dictionary<int, float>();

        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && contextIndices.Count >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictionsFromIndices(candidates, contextIndices, n, weight);
            }
        }

        if (contextIndices.Count > 0)
        {
            int lastIdx = contextIndices[contextIndices.Count - 1];
            int nodeIndex = FindNodeBySequenceIndex(lastIdx);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(10);
                foreach (var (index, score) in topNext)
                {
                    if (!candidates.ContainsKey(index))
                        candidates[index] = 0;
                    candidates[index] += score * 1.0f;
                }
            }
        }

        if (candidates.Count == 0)
            return null;

        if (boostRare)
        {
            float maxScore = candidates.Values.Max();
            var boostedCandidates = new Dictionary<int, float>();
            foreach (var kvp in candidates)
            {
                float normalizedScore = kvp.Value / maxScore;
                float boostedScore = (float)Math.Pow(normalizedScore, 0.7);
                boostedCandidates[kvp.Key] = boostedScore;
            }
            candidates = boostedCandidates;
        }

        var scaledScores = ApplyTemperatureSoftmax(candidates);
        int selectedIndex = SampleFromDistribution(scaledScores);

        if (selectedIndex >= 0 && selectedIndex < fullSequence.Count)
            return fullSequence[selectedIndex].ToTensor();

        return null;
    }

    /// <summary>
    /// Get diverse predictions by sampling multiple times
    /// </summary>
    public List<(Tensor tensor, int frequency)> GetDiversePredictions(Tensor[] context, int samples = 10)
    {
        var indexFrequency = new Dictionary<int, int>();

        List<int> contextIndices = new List<int>();
        foreach (var tensor in context)
        {
            int idx = FindSimilarTensorIndex(tensor);
            if (idx >= 0)
                contextIndices.Add(idx);
        }

        for (int i = 0; i < samples; i++)
        {
            var sampled = SampleNextStochasticIndex(contextIndices, boostRare: true);
            if (sampled >= 0)
            {
                if (!indexFrequency.ContainsKey(sampled))
                    indexFrequency[sampled] = 0;
                indexFrequency[sampled]++;
            }
        }

        return indexFrequency
            .OrderByDescending(kvp => kvp.Value)
            .Where(kvp => kvp.Key >= 0 && kvp.Key < fullSequence.Count)
            .Select(kvp => (fullSequence[kvp.Key].ToTensor(), kvp.Value))
            .ToList();
    }

    /// <summary>
    /// Continue the learned sequence (generate new data based on what was learned)
    /// </summary>
    public Tensor[] ContinueSequence(int count = 10, bool useBlending = false)
    {
        if (fullSequence.Count == 0)
            return new Tensor[0];

        int contextSize = Math.Min(maxContextWindow - 1, fullSequence.Count);
        Tensor[] context = new Tensor[contextSize];

        for (int i = 0; i < contextSize; i++)
        {
            int idx = fullSequence.Count - contextSize + i;
            context[i] = fullSequence[idx].ToTensor();
        }

        return PredictNext(context, count, useBlending);
    }

    /// <summary>
    /// Find similar tensors in the learned sequence
    /// </summary>
    public List<(Tensor tensor, float similarity)> GetSimilarTensors(Tensor queryTensor, int topN = 5)
    {
        var similarities = new List<(int, float)>();
        QuantizedTensor qQuery = QuantizedTensor.FromTensor(queryTensor);

        for (int i = 0; i < fullSequence.Count; i++)
        {
            float similarity = qQuery.CosineSimilarityQuantized(fullSequence[i]);
            similarities.Add((i, similarity));
        }

        return similarities
            .OrderByDescending(s => s.Item2)
            .Take(topN)
            .Select(s => (fullSequence[s.Item1].ToTensor(), s.Item2))
            .ToList();
    }

    /// <summary>
    /// Interpolate between two tensors
    /// </summary>
    public Tensor Interpolate(Tensor from, Tensor to, float t)
    {
        if (from.Size != to.Size)
            throw new ArgumentException("Tensors must have same size");

        Tensor result = new Tensor(from.Size);
        for (int i = 0; i < from.Size; i++)
        {
            result.Data[i] = from.Data[i] * (1 - t) + to.Data[i] * t;
        }

        return result;
    }

    /// <summary>
    /// Predict next tensor using regression (generates novel tensors, not just retrieval)
    /// This creates new tensors by weighted averaging of likely next states
    /// </summary>
    public Tensor PredictNextRegression(Tensor[] context, float noveltyBias = 0.2f)
    {
        if (context == null || context.Length == 0 || fullSequence.Count == 0)
            return null;

        // Convert context to indices
        List<int> contextIndices = new List<int>();
        foreach (var tensor in context)
        {
            int idx = FindSimilarTensorIndex(tensor);
            if (idx >= 0)
                contextIndices.Add(idx);
        }

        if (contextIndices.Count == 0)
            return null;

        // Gather all candidate next tensors with their weights
        var weightedCandidates = new Dictionary<int, float>();

        // From n-grams
        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && contextIndices.Count >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictionsFromIndices(weightedCandidates, contextIndices, n, weight);
            }
        }

        // From node graph
        if (contextIndices.Count > 0)
        {
            int lastIdx = contextIndices[contextIndices.Count - 1];
            int nodeIndex = FindNodeBySequenceIndex(lastIdx);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(10);
                foreach (var (index, score) in topNext)
                {
                    if (!weightedCandidates.ContainsKey(index))
                        weightedCandidates[index] = 0;
                    weightedCandidates[index] += score * 2.0f;
                }
            }
        }

        if (weightedCandidates.Count == 0)
            return null;

        // Normalize weights to probabilities
        float totalWeight = weightedCandidates.Values.Sum();
        if (totalWeight == 0)
            return null;

        // Create the regressed tensor through weighted averaging
        Tensor result = new Tensor(tensorSize);
        Array.Fill(result.Data, 0f);

        foreach (var kvp in weightedCandidates)
        {
            int candidateIdx = kvp.Key;
            float probability = kvp.Value / totalWeight;

            if (candidateIdx >= 0 && candidateIdx < fullSequence.Count)
            {
                float[] candidateData = fullSequence[candidateIdx].Dequantize();

                for (int i = 0; i < tensorSize; i++)
                {
                    result.Data[i] += candidateData[i] * probability;
                }
            }
        }

        // Apply novelty bias: add small random perturbation to generate truly novel tensors
        if (noveltyBias > 0)
        {
            for (int i = 0; i < tensorSize; i++)
            {
                float perturbation = ((float)random.NextDouble() - 0.5f) * 2f * noveltyBias;
                result.Data[i] += perturbation * result.Data[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Predict multiple next tensors using regression
    /// Each prediction builds on the previous regressed tensors
    /// </summary>
    public Tensor[] PredictMultipleRegression(Tensor[] context, int count = 1, float noveltyBias = 0.2f)
    {
        if (context == null || context.Length == 0 || count <= 0)
            return new Tensor[0];

        List<Tensor> predictions = new List<Tensor>();
        List<Tensor> currentContext = new List<Tensor>(context);

        for (int i = 0; i < count; i++)
        {
            Tensor next = PredictNextRegression(currentContext.ToArray(), noveltyBias);

            if (next == null)
                break;

            predictions.Add(next);

            // Update context window for next prediction
            currentContext.Add(next);
            if (currentContext.Count > maxContextWindow)
                currentContext.RemoveAt(0);
        }

        return predictions.ToArray();
    }

    /// <summary>
    /// Advanced regression that considers temporal trends and extrapolation
    /// Analyzes the delta/change pattern in context to extrapolate future states
    /// </summary>
    public Tensor PredictNextRegressionWithTrend(Tensor[] context, float trendWeight = 0.5f, float noveltyBias = 0.1f)
    {
        if (context == null || context.Length < 2 || fullSequence.Count == 0)
            return PredictNextRegression(context, noveltyBias);

        // First get the base regression prediction
        Tensor baseRegression = PredictNextRegression(context, 0f); // No novelty yet

        if (baseRegression == null)
            return null;

        // Calculate the average trend/delta in the context sequence
        Tensor trendVector = new Tensor(tensorSize);
        Array.Fill(trendVector.Data, 0f);

        int trendSamples = 0;
        for (int i = 1; i < context.Length; i++)
        {
            for (int j = 0; j < tensorSize; j++)
            {
                float delta = context[i].Data[j] - context[i - 1].Data[j];
                trendVector.Data[j] += delta;
            }
            trendSamples++;
        }

        // Average the trend
        if (trendSamples > 0)
        {
            for (int i = 0; i < tensorSize; i++)
            {
                trendVector.Data[i] /= trendSamples;
            }
        }

        // Combine base regression with trend extrapolation
        Tensor result = new Tensor(tensorSize);
        for (int i = 0; i < tensorSize; i++)
        {
            float baseValue = baseRegression.Data[i];
            float trendValue = context[context.Length - 1].Data[i] + trendVector.Data[i];

            // Weighted blend of regression and trend extrapolation
            result.Data[i] = baseValue * (1 - trendWeight) + trendValue * trendWeight;

            // Add novelty perturbation
            if (noveltyBias > 0)
            {
                float perturbation = ((float)random.NextDouble() - 0.5f) * 2f * noveltyBias;
                result.Data[i] += perturbation * Math.Abs(result.Data[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Hybrid prediction: combines retrieval and regression
    /// Returns both the most likely retrieved tensor and a regressed novel tensor
    /// </summary>
    public (Tensor retrieved, Tensor regressed) PredictNextHybrid(Tensor[] context, float noveltyBias = 0.15f)
    {
        Tensor retrieved = null;
        Tensor regressed = null;

        // Get retrieval prediction
        var retrievedArray = PredictNext(context, 1, useBlending: false, useStochastic: false);
        if (retrievedArray != null && retrievedArray.Length > 0)
        {
            retrieved = retrievedArray[0];
        }

        // Get regression prediction
        regressed = PredictNextRegression(context, noveltyBias);

        return (retrieved, regressed);
    }

    /// <summary>
    /// Generate a completely novel sequence using regression
    /// Useful for creative generation that builds on learned patterns but creates new content
    /// </summary>
    public Tensor[] GenerateNovelSequence(Tensor[] seed, int length = 10, float noveltyBias = 0.2f, float trendWeight = 0.3f)
    {
        if (seed == null || seed.Length == 0 || length <= 0)
            return new Tensor[0];

        List<Tensor> sequence = new List<Tensor>(seed);

        for (int i = 0; i < length; i++)
        {
            // Get context window
            int contextSize = Math.Min(maxContextWindow - 1, sequence.Count);
            Tensor[] context = sequence.Skip(sequence.Count - contextSize).ToArray();

            // Use trend-aware regression for more coherent generation
            Tensor next = PredictNextRegressionWithTrend(context, trendWeight, noveltyBias);

            if (next == null)
                break;

            sequence.Add(next);
        }

        // Return only the newly generated portion
        return sequence.Skip(seed.Length).ToArray();
    }

    private int FindSimilarTensorIndex(Tensor tensor)
    {
        QuantizedTensor qTensor = QuantizedTensor.FromTensor(tensor);

        int hash = ComputeTensorHash(qTensor);
        if (tensorHashIndex.TryGetValue(hash, out int idx))
        {
            if (idx >= 0 && idx < fullSequence.Count)
            {
                float sim = qTensor.CosineSimilarityQuantized(fullSequence[idx]);
                if (sim >= 0.90f)  // FIXED: More lenient threshold for matching predictions
                    return idx;
            }
        }

        // Fallback: search ALL entries if needed for exact matches
        for (int i = 0; i < fullSequence.Count; i++)
        {
            float sim = qTensor.CosineSimilarityQuantized(fullSequence[i]);
            if (sim >= 0.90f)  // FIXED: More lenient threshold
                return i;
        }

        return -1;
    }

    private int PredictSingleNextIndex(List<int> contextIndices, bool useBlending)
    {
        var candidates = new Dictionary<int, float>();

        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && contextIndices.Count >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictionsFromIndices(candidates, contextIndices, n, weight);
            }
        }

        if (contextIndices.Count > 0)
        {
            int lastIdx = contextIndices[contextIndices.Count - 1];
            int nodeIndex = FindNodeBySequenceIndex(lastIdx);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(10);
                foreach (var (index, score) in topNext)
                {
                    if (!candidates.ContainsKey(index))
                        candidates[index] = 0;
                    candidates[index] += score * 2.0f;  // FIXED: Boost node predictions more
                }
            }
        }

        if (candidates.Count == 0)
            return -1;

        return candidates.OrderByDescending(kvp => kvp.Value).First().Key;
    }

    private int SampleNextStochasticIndex(List<int> contextIndices, bool boostRare)
    {
        var candidates = new Dictionary<int, float>();

        if (random.NextDouble() < explorationRate && fullSequence.Count > 0)
        {
            return random.Next(fullSequence.Count);
        }

        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && contextIndices.Count >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictionsFromIndices(candidates, contextIndices, n, weight);
            }
        }

        if (contextIndices.Count > 0)
        {
            int lastIdx = contextIndices[contextIndices.Count - 1];
            int nodeIndex = FindNodeBySequenceIndex(lastIdx);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(10);
                foreach (var (index, score) in topNext)
                {
                    if (!candidates.ContainsKey(index))
                        candidates[index] = 0;
                    candidates[index] += score;
                }
            }
        }

        if (candidates.Count == 0)
            return -1;

        var scaledScores = ApplyTemperatureSoftmax(candidates);
        return SampleFromDistribution(scaledScores);
    }

    private void TryAddPredictionsFromIndices(Dictionary<int, float> candidates, List<int> contextIndices, int n, float weight)
    {
        int contextSize = n - 1;
        if (contextIndices.Count < contextSize) return;

        List<int> contextWindow = contextIndices
            .Skip(contextIndices.Count - contextSize)
            .Take(contextSize)
            .ToList();

        int contextHash = ComputeContextHashFromIndices(contextWindow);

        if (nGramHashIndex[n].TryGetValue(contextHash, out int entryIndex))
        {
            var entry = nGrams[n][entryIndex];
            if (ContextIndicesMatch(entry.contextIndices, contextWindow))
            {
                float total = entry.nextIndices.Values.Sum();
                foreach (var kvp in entry.nextIndices)
                {
                    if (!candidates.ContainsKey(kvp.Key))
                        candidates[kvp.Key] = 0;

                    float probability = kvp.Value / total;
                    candidates[kvp.Key] += probability * weight;
                }
            }
        }
    }

    private int FindNodeBySequenceIndex(int seqIdx)
    {
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].SequenceIndex == seqIdx)
                return i;
        }
        return -1;
    }

    private Dictionary<int, float> ApplyTemperatureSoftmax(Dictionary<int, float> scores)
    {
        var scaledScores = new Dictionary<int, float>();
        float maxScore = scores.Values.Max();

        var expScores = new Dictionary<int, float>();
        float sumExp = 0;

        foreach (var kvp in scores)
        {
            float scaledScore = (kvp.Value - maxScore) / temperature;
            float expScore = (float)Math.Exp(scaledScore);
            expScores[kvp.Key] = expScore;
            sumExp += expScore;
        }

        foreach (var kvp in expScores)
        {
            scaledScores[kvp.Key] = kvp.Value / sumExp;
        }

        return scaledScores;
    }

    private int SampleFromDistribution(Dictionary<int, float> probabilities)
    {
        float rand = (float)random.NextDouble();
        float cumulative = 0;

        foreach (var kvp in probabilities)
        {
            cumulative += kvp.Value;
            if (rand <= cumulative)
                return kvp.Key;
        }

        return probabilities.Keys.Last();
    }

    private void PruneIfNeeded()
    {
        if (nodes.Count > MAX_NODES)
        {
            var sortedNodes = nodes
                .Select((node, idx) => (node, idx, transitionCount: node.GetNextProbabilities().Count))
                .OrderByDescending(x => x.transitionCount)
                .ToList();

            int toKeep = (int)(MAX_NODES * 0.8f);
            nodes = sortedNodes.Take(toKeep).Select(x => x.node).ToList();
        }

        foreach (var n in nGrams.Keys.ToList())
        {
            if (nGrams[n].Count > MAX_NGRAM_ENTRIES_PER_N)
            {
                var sorted = nGrams[n]
                    .OrderByDescending(entry => entry.nextIndices.Values.Sum())
                    .Take((int)(MAX_NGRAM_ENTRIES_PER_N * 0.8f))
                    .ToList();
                nGrams[n] = sorted;

                nGramHashIndex[n].Clear();
                for (int i = 0; i < sorted.Count; i++)
                {
                    int hash = ComputeContextHashFromIndices(sorted[i].contextIndices);
                    nGramHashIndex[n][hash] = i;
                }
            }
        }
    }

    private void RebuildTensorHashIndex()
    {
        tensorHashIndex.Clear();
        for (int i = 0; i < fullSequence.Count; i++)
        {
            int hash = ComputeTensorHash(fullSequence[i]);
            tensorHashIndex[hash] = i;
        }
    }

    public string GetStatistics()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine($"Learned sequence length: {fullSequence.Count} tensors");
        sb.AppendLine($"Unique tensor nodes: {nodes.Count} / {MAX_NODES} max");
        sb.AppendLine($"Tensor size: {tensorSize}");
        sb.AppendLine($"Quantization: {(useQuantization ? "Enabled" : "Disabled")}");
        sb.AppendLine($"Delta Regression: {(useDeltaRegression ? "Enabled" : "Disabled")}");
        sb.AppendLine($"Similarity threshold: {similarityThreshold:F3}");
        sb.AppendLine($"Temperature: {temperature:F2}");
        sb.AppendLine($"Exploration rate: {explorationRate:F3}");

        long quantizedBytes = fullSequence.Count * (tensorSize + 8 + 16);
        long unquantizedBytes = fullSequence.Count * (tensorSize * 4 + 16);
        long nodeBytes = nodes.Count * 24;
        long ngramBytes = nGrams.Sum(kvp => kvp.Value.Count * (kvp.Key * 4 + 64));
        long deltaBytes = useDeltaRegression ? (tensorSize * 4 * 4) + (deltaHistory.Count * tensorSize * 4 * 4) : 0;
        long estimatedMemory = quantizedBytes + nodeBytes + ngramBytes + deltaBytes;

        sb.AppendLine($"\nMemory (optimized): {estimatedMemory / 1024.0 / 1024.0:F2} MB");
        sb.AppendLine($"  FullSequence: {quantizedBytes / 1024.0 / 1024.0:F2} MB");
        sb.AppendLine($"  Nodes: {nodeBytes / 1024.0 / 1024.0:F2} MB");
        sb.AppendLine($"  N-grams: {ngramBytes / 1024.0 / 1024.0:F2} MB");
        if (useDeltaRegression)
            sb.AppendLine($"  Delta States: {deltaBytes / 1024.0 / 1024.0:F2} MB");

        if (useQuantization)
        {
            float savings = (1 - (float)quantizedBytes / unquantizedBytes) * 100;
            sb.AppendLine($"\nSavings from quantization: ~{savings:F1}%");
        }

        // Graph statistics
        int totalTransitions = nodes.Sum(n => n.GetNextIndices().Count);
        int sharedDestinations = CountSharedDestinations();
        sb.AppendLine($"\nGraph Statistics:");
        sb.AppendLine($"  Total transitions: {totalTransitions}");
        sb.AppendLine($"  Shared destinations (cross-pollination): {sharedDestinations}");

        // Delta regression statistics
        if (useDeltaRegression && deltaState != null)
        {
            sb.AppendLine($"\nDelta Regression Statistics:");
            sb.AppendLine($"  Step count: {deltaState.StepCount}");
            sb.AppendLine($"  Prediction confidence: {deltaState.GetPredictionConfidence():F3}");
            sb.AppendLine($"  Delta history size: {deltaHistory.Count}");

            if (deltaHistory.Count >= 2)
            {
                var (avgVel, trend) = AnalyzeDeltaTrends();
                sb.AppendLine($"  Average velocity: {avgVel:F4}");
                sb.AppendLine($"  Trend direction: {trend:F4}");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Count how many tensor indices are pointed to by multiple different nodes
    /// This indicates the level of cross-tree pollination (graph connectivity)
    /// </summary>
    private int CountSharedDestinations()
    {
        var destinationSources = new Dictionary<int, HashSet<int>>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var nextIndices = nodes[i].GetNextIndices();
            foreach (var destIdx in nextIndices.Keys)
            {
                if (!destinationSources.ContainsKey(destIdx))
                    destinationSources[destIdx] = new HashSet<int>();
                destinationSources[destIdx].Add(i);
            }
        }

        return destinationSources.Count(kvp => kvp.Value.Count > 1);
    }

    private int ComputeTensorHash(QuantizedTensor tensor)
    {
        unchecked
        {
            int hash = 17;
            for (int i = 0; i < tensor.QuantizedData.Length; i += 2)
            {
                hash = hash * 31 + tensor.QuantizedData[i];
            }
            return hash;
        }
    }

    private int ComputeContextHashFromIndices(List<int> indices)
    {
        unchecked
        {
            int hash = 17;
            foreach (int idx in indices)
            {
                if (idx >= 0 && idx < fullSequence.Count)
                {
                    hash = hash * 31 + ComputeTensorHash(fullSequence[idx]);
                }
            }
            return hash;
        }
    }

    private float GetTemporalWeight()
    {
        return (float)Math.Pow(temporalDecayFactor, transitionCounter);
    }

    private void AdaptSimilarityThreshold()
    {
        if (nodes.Count > MAX_NODES * 0.9f)
        {
            similarityThreshold = Math.Max(0.93f, baseSimilarityThreshold - 0.02f);
        }
        else if (nodes.Count > MAX_NODES * 0.8f)
        {
            similarityThreshold = Math.Max(0.94f, baseSimilarityThreshold - 0.01f);
        }
        else
        {
            similarityThreshold = baseSimilarityThreshold;
        }
    }

    /// <summary>
    /// Learn from multiple sequences in a batch with improved efficiency
    /// </summary>
    public void LearnBatch(List<(Tensor[] sequence, float outcome)> batch)
    {
        if (batch == null || batch.Count == 0) return;

        foreach (var (sequence, outcome) in batch)
        {
            LearnWithOutcome(sequence, outcome);
        }

        // Perform batch optimization if using experience replay
        if (useExperienceReplay && experienceBuffer.Count >= 16)
        {
            ReplayExperiences(batchSize: 16, epochs: 1);
        }
    }

    /// <summary>
    /// Replay experiences from buffer with prioritization
    /// </summary>
    public void ReplayExperiences(int batchSize = 32, int epochs = 1)
    {
        if (!useExperienceReplay || experienceBuffer.Count == 0) return;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Sample prioritized batch
            var batch = SamplePrioritizedBatch(batchSize);

            foreach (var (sequence, outcome, _) in batch)
            {
                LearnInternal(sequence, outcome * 0.5f); // Reduced weight for replay to avoid overfitting
            }
        }
    }

    private List<(Tensor[] sequence, float outcome, long timestamp)> SamplePrioritizedBatch(int batchSize)
    {
        if (experienceBuffer.Count <= batchSize)
            return experienceBuffer.ToList();

        // Prioritized sampling: higher probability for high-outcome experiences
        var priorities = experienceBuffer.Select(exp =>
        {
            float outcomePriority = Math.Abs(exp.outcome) + 0.1f; // Boost extreme outcomes
            float recencyPriority = 1.0f + (exp.timestamp / (float)transitionCounter) * 0.5f; // Boost recent
            return outcomePriority * recencyPriority;
        }).ToArray();

        float totalPriority = priorities.Sum();
        var batch = new List<(Tensor[], float, long)>();

        for (int i = 0; i < batchSize; i++)
        {
            float rand = (float)random.NextDouble() * totalPriority;
            float cumulative = 0;

            for (int j = 0; j < experienceBuffer.Count; j++)
            {
                cumulative += priorities[j];
                if (rand <= cumulative)
                {
                    batch.Add(experienceBuffer[j]);
                    break;
                }
            }
        }

        return batch;
    }

    public void Clear()
    {
        nodes.Clear();
        fullSequence.Clear();
        foreach (var dict in nGrams.Values)
        {
            dict.Clear();
        }
        nGramHashIndex.Clear();
        tensorHashIndex.Clear();
        experienceBuffer.Clear();
        deltaHistory.Clear();
        deltaState = null;
        tensorSize = 0;
        transitionCounter = 0;
        similarityThreshold = baseSimilarityThreshold;
    }

    public void SetTemperature(float temp) => temperature = Math.Max(0.1f, temp);
    public void SetExplorationRate(float rate) => explorationRate = Math.Clamp(rate, 0f, 1f);
    public float GetTemperature() => temperature;
    public float GetExplorationRate() => explorationRate;

    // NEW: Delta regression configuration
    public void SetDeltaMomentum(float momentum)
    {
        if (deltaState != null)
            deltaState.Momentum = Math.Clamp(momentum, 0f, 1f);
    }

    public void SetDeltaSmoothingFactor(float alpha)
    {
        if (deltaState != null)
            deltaState.SmoothingFactor = Math.Clamp(alpha, 0f, 1f);
    }
}