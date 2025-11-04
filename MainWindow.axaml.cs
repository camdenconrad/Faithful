using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NNImage;

public partial class MainWindow : Window
{
    private AdjacencyGraph? _adjacencyGraph;
    private List<string> _trainingImagePaths = new();
    private Bitmap? _generatedBitmap;

    public MainWindow()
    {
        InitializeComponent();
    }

    private async void SelectFolderButton_Click(object? sender, RoutedEventArgs e)
    {
        var folder = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select Image Folder",
            AllowMultiple = false
        });

        if (folder.Count > 0)
        {
            var folderPath = folder[0].Path.LocalPath;
            FolderPathTextBox.Text = folderPath;

            // Load sample images
            LoadSampleImages(folderPath);
        }
    }

    private void LoadSampleImages(string folderPath)
    {
        SampleImagesPanel.Children.Clear();
        _trainingImagePaths.Clear();

        var supportedExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
        var imageFiles = Directory.GetFiles(folderPath)
            .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLower()))
            .Take(20)
            .ToList();

        _trainingImagePaths = imageFiles;

        foreach (var imagePath in imageFiles.Take(6))
        {
            try
            {
                var bitmap = new Bitmap(imagePath);
                var image = new Image
                {
                    Source = bitmap,
                    Width = 120,
                    Height = 120,
                    Margin = new Avalonia.Thickness(5),
                    Stretch = Avalonia.Media.Stretch.Uniform
                };
                SampleImagesPanel.Children.Add(image);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading image {imagePath}: {ex.Message}");
            }
        }

        TrainingStatusText.Text = $"Found {_trainingImagePaths.Count} images";
        TrainingStatusText.Foreground = Avalonia.Media.Brushes.Green;
    }

    private async void TrainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_trainingImagePaths.Count == 0)
        {
            TrainingStatusText.Text = "No images selected";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Red;
            return;
        }

        TrainButton.IsEnabled = false;
        GenerateButton.IsEnabled = false;
        TrainingStatusText.Text = "Training...";
        TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

        await Task.Run(() =>
        {
            try
            {
                var quantizationLevel = (int)QuantizationLevel.Value;
                var trainer = new ImageTrainer(quantizationLevel);

                int processedCount = 0;
                foreach (var imagePath in _trainingImagePaths)
                {
                    trainer.ProcessImage(imagePath);
                    processedCount++;

                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        TrainingStatusText.Text = $"Processing {processedCount}/{_trainingImagePaths.Count}...";
                    });
                }

                _adjacencyGraph = trainer.GetAdjacencyGraph();

                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    TrainingStatusText.Text = $"Training complete! {_adjacencyGraph.GetColorCount()} unique colors";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.Green;
                    GenerateButton.IsEnabled = true;
                });
            }
            catch (Exception ex)
            {
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    TrainingStatusText.Text = $"Error: {ex.Message}";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.Red;
                });
            }
            finally
            {
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    TrainButton.IsEnabled = true;
                });
            }
        });
    }

    private async void GenerateButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_adjacencyGraph == null)
        {
            TrainingStatusText.Text = "Please train the model first";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Red;
            return;
        }

        GenerateButton.IsEnabled = false;
        SaveButton.IsEnabled = false;
        TrainingStatusText.Text = "Generating image...";
        TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

        await Task.Run(() =>
        {
            try
            {
                var width = (int)OutputWidth.Value;
                var height = (int)OutputHeight.Value;

                var wfc = new WaveFunctionCollapse(_adjacencyGraph, width, height);
                var generatedImage = wfc.Generate();

                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    _generatedBitmap = generatedImage;
                    GeneratedImage.Source = generatedImage;
                    GeneratedImage.Width = width;
                    GeneratedImage.Height = height;

                    TrainingStatusText.Text = "Image generated successfully!";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.Green;
                    SaveButton.IsEnabled = true;
                });
            }
            catch (Exception ex)
            {
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    TrainingStatusText.Text = $"Generation error: {ex.Message}";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.Red;
                });
            }
            finally
            {
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    GenerateButton.IsEnabled = true;
                });
            }
        });
    }

    private async void SaveButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_generatedBitmap == null)
            return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save Generated Image",
            DefaultExtension = "png",
            SuggestedFileName = $"generated_{DateTime.Now:yyyyMMdd_HHmmss}.png",
            FileTypeChoices = new[]
            {
                new FilePickerFileType("PNG Image") { Patterns = new[] { "*.png" } }
            }
        });

        if (file != null)
        {
            using var stream = await file.OpenWriteAsync();
            _generatedBitmap.Save(stream);

            TrainingStatusText.Text = "Image saved successfully!";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Green;
        }
    }
}
