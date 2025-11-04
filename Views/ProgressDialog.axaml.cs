using Avalonia.Controls;
using Avalonia.Threading;
using System;
using System.Text;

namespace NNImage.Views;

public partial class ProgressDialog : Window
{
    private readonly StringBuilder _logBuilder = new();

    public ProgressDialog()
    {
        InitializeComponent();
    }

    public void UpdateProgress(int current, int total, string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var percentage = total > 0 ? (current * 100.0 / total) : 0;
            ProgressBar.Value = percentage;
            StatusText.Text = message;

            var logMessage = $"[{DateTime.Now:HH:mm:ss}] {message}";
            _logBuilder.AppendLine(logMessage);
            LogText.Text = _logBuilder.ToString();

            Console.WriteLine(logMessage);
        });
    }

    public void SetTitle(string title)
    {
        Dispatcher.UIThread.Post(() =>
        {
            TitleText.Text = title;
        });
    }

    public void AddLog(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var logMessage = $"[{DateTime.Now:HH:mm:ss}] {message}";
            _logBuilder.AppendLine(logMessage);
            LogText.Text = _logBuilder.ToString();
            Console.WriteLine(logMessage);
        });
    }

    public void Complete(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            ProgressBar.Value = 100;
            StatusText.Text = message;
            StatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
            AddLog($"✓ {message}");
        });
    }

    public void Error(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            StatusText.Text = message;
            StatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            AddLog($"✗ ERROR: {message}");
        });
    }
}
