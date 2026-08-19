using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using AI_Visual_Quality_Controller.Properties;

namespace AI_Visual_Quality_Controller;

public class Form1 : Form
{
	private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

	public struct RECT
	{
		public int Left;

		public int Top;

		public int Right;

		public int Bottom;
	}

	private List<string> accumulatedText = new List<string>();

	private Process darknetProcess;

	private bool isRunning;

	private IContainer components = null;

	private SplitContainer splitContainer1;

	private SplitContainer splitContainer2;

	private Panel camOutput;

	private TextBox txtConsoleOutput;

	private Button btnStartStop;

	private Label lblFPS;

	private Label lblAvgFPS;

	private TrackBar trackBarF_Body;

	private Label label3;

	private Label label2;

	private Label label1;

	private TrackBar trackBarRed_T;

	private TrackBar trackBarCut;

	private Label lblCut;

	private Label lblRed_T;

	private Label lblF_Body;

	private Label label7;

	private Label label4;

	private Label label5;

	private Label label6;

	private Label lblDressing;

	private Panel panelRedLight;

	private PictureBox pictureBoxDarknet;

	private Timer timerUpdateDarknetWindow;

	private Label label9;

	private Label label8;

	private GroupBox groupBox1;

	private Label label11;

	private Label lblDPS;

	private Label label10;

	private GroupBox groupBox3;

	private GroupBox groupBox2;

	private Label lblThrCut;

	private Label lblThrRtape;

	private Label lblThrFbody;

	private Label label12;

	[DllImport("user32.dll", SetLastError = true)]
	[return: MarshalAs(UnmanagedType.Bool)]
	private static extern bool PrintWindow(IntPtr hwnd, IntPtr hDC, uint nFlags);

	[DllImport("user32.dll", SetLastError = true)]
	public static extern bool GetWindowRect(IntPtr hwnd, out RECT lpRect);

	[DllImport("user32.dll")]
	[return: MarshalAs(UnmanagedType.Bool)]
	private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

	[DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
	private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

	public Form1()
	{
		InitializeComponent();
		pictureBoxDarknet.BackgroundImage = Resources.AIVQC_logos;
		timerUpdateDarknetWindow.Interval = 100;
		timerUpdateDarknetWindow.Tick += timerUpdateDarknetWindow_Tick;
		timerUpdateDarknetWindow.Enabled = true;
		lblThrFbody.Text = trackBarF_Body.Value + "%";
		lblThrRtape.Text = trackBarRed_T.Value + "%";
		lblThrCut.Text = trackBarCut.Value + "%";
	}

	private void btnStartStop_Click(object sender, EventArgs e)
	{
		if (!isRunning)
		{
			StartDarknet();
		}
		else
		{
			StopDarknet();
		}
	}

	private void StartDarknet()
	{
		string text = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Darknet");
		string fileName = Path.Combine(text, "darknet.exe");
		ProcessStartInfo startInfo = new ProcessStartInfo
		{
			FileName = fileName,
			Arguments = "detector demo obj.data custom-yolov4-tiny-detector_4class_416.cfg custom-yolov4-tiny-detector_4class_416_bestv3.weights",
			UseShellExecute = false,
			RedirectStandardOutput = true,
			CreateNoWindow = true,
			WorkingDirectory = text
		};
		darknetProcess = new Process
		{
			StartInfo = startInfo,
			EnableRaisingEvents = true
		};
		darknetProcess.OutputDataReceived += DarknetProcess_OutputDataReceived;
		darknetProcess.Exited += DarknetProcess_Exited;
		darknetProcess.Start();
		darknetProcess.BeginOutputReadLine();
		btnStartStop.Text = "Stop Detection";
		btnStartStop.BackColor = Color.IndianRed;
		isRunning = true;
		timerUpdateDarknetWindow.Start();
	}

	private void StopDarknet()
	{
		if (darknetProcess != null && !darknetProcess.HasExited)
		{
			darknetProcess.Kill();
		}
		timerUpdateDarknetWindow.Stop();
	}

	private void DarknetProcess_OutputDataReceived(object sender, DataReceivedEventArgs e)
	{
		if (string.IsNullOrEmpty(e.Data))
		{
			return;
		}
		Invoke((Action)delegate
		{
			txtConsoleOutput.AppendText(e.Data + Environment.NewLine);
		});
		if (e.Data.StartsWith("FPS:"))
		{
			int num = e.Data.IndexOf("FPS:");
			int num2 = e.Data.IndexOf("AVG_FPS:");
			string fps = e.Data.Substring(num + "FPS:".Length, num2 - (num + "FPS:".Length)).Trim();
			string avgFps = e.Data.Substring(num2 + "AVG_FPS:".Length).Trim();
			Invoke((Action)delegate
			{
				lblFPS.Text = fps;
				lblAvgFPS.Text = avgFps;
				int num3 = (int)Convert.ToDouble(avgFps);
				lblDPS.Text = (num3 * 60).ToString();
			});
			ProcessAccumulatedText();
			accumulatedText.Clear();
		}
		else
		{
			accumulatedText.Add(e.Data);
		}
	}

	private void ProcessAccumulatedText()
	{
		if (!accumulatedText.Any((string line) => line.StartsWith("Objects:")))
		{
			return;
		}
		bool flag = false;
		List<string> list = new List<string>();
		foreach (string item in accumulatedText)
		{
			if (item.StartsWith("Objects:"))
			{
				flag = true;
			}
			else if (flag)
			{
				string text = item.Trim();
				if (text.EndsWith("%"))
				{
					list.Add(text);
				}
				else
				{
					flag = false;
				}
			}
		}
		Invoke((Action)delegate
		{
			lblCut.Text = "0%";
			lblDressing.Text = "0%";
			lblF_Body.Text = "0%";
			lblRed_T.Text = "0%";
		});
		bool redLightOn = false;
		foreach (string item2 in list)
		{
			string[] array = item2.Split(new char[1] { ':' }, StringSplitOptions.RemoveEmptyEntries);
			if (array.Length != 2)
			{
				continue;
			}
			string text2 = array[0].Trim();
			string percentage = array[1].Trim();
			int threshold = 0;
			switch (text2)
			{
			case "Cut":
				Invoke((Action)delegate
				{
					threshold = trackBarCut.Value;
				});
				Invoke((Action)delegate
				{
					lblCut.Text = percentage;
				});
				break;
			case "Dressing":
				Invoke((Action)delegate
				{
					lblDressing.Text = percentage;
				});
				break;
			case "F_Body":
				Invoke((Action)delegate
				{
					threshold = trackBarF_Body.Value;
				});
				Invoke((Action)delegate
				{
					lblF_Body.Text = percentage;
				});
				break;
			case "Red_T":
				Invoke((Action)delegate
				{
					threshold = trackBarRed_T.Value;
				});
				Invoke((Action)delegate
				{
					lblRed_T.Text = percentage;
				});
				break;
			}
			int num = int.Parse(percentage.TrimEnd('%'));
			if (text2 != "Dressing" && num > threshold)
			{
				redLightOn = true;
			}
		}
		Invoke((Action)delegate
		{
			panelRedLight.Visible = redLightOn;
		});
	}

	private void DarknetProcess_Exited(object sender, EventArgs e)
	{
		Invoke((Action)delegate
		{
			btnStartStop.Text = "Start Detection";
			btnStartStop.BackColor = Color.LimeGreen;
			isRunning = false;
		});
	}

	private IntPtr FindDarknetWindow()
	{
		IntPtr hWnd = IntPtr.Zero;
		string windowTitle = "Demo";
		EnumWindows(delegate(IntPtr hWndEnum, IntPtr lParam)
		{
			StringBuilder stringBuilder = new StringBuilder(256);
			GetWindowText(hWndEnum, stringBuilder, 256);
			if (stringBuilder.ToString().Contains(windowTitle))
			{
				hWnd = hWndEnum;
				return false;
			}
			return true;
		}, IntPtr.Zero);
		return hWnd;
	}

	private Bitmap CaptureDarknetWindow(IntPtr hwnd)
	{
		GetWindowRect(hwnd, out var lpRect);
		int num = lpRect.Right - lpRect.Left;
		int num2 = lpRect.Bottom - lpRect.Top;
		Bitmap image = new Bitmap(num, num2, PixelFormat.Format32bppArgb);
		using (Graphics graphics = Graphics.FromImage(image))
		{
			PrintWindow(hwnd, graphics.GetHdc(), 0u);
			graphics.ReleaseHdc();
		}
		float num3 = Math.Min((float)camOutput.Width / (float)num, (float)camOutput.Height / (float)num2);
		int num4 = (int)((float)num * num3);
		int num5 = (int)((float)num2 * num3);
		Bitmap bitmap = new Bitmap(num4, num5, PixelFormat.Format32bppArgb);
		using (Graphics graphics2 = Graphics.FromImage(bitmap))
		{
			graphics2.DrawImage(image, 0, 0, num4, num5);
		}
		return bitmap;
	}

	private void timerUpdateDarknetWindow_Tick(object sender, EventArgs e)
	{
		IntPtr intPtr = FindDarknetWindow();
		if (intPtr != IntPtr.Zero)
		{
			Bitmap bitmap = CaptureDarknetWindow(intPtr);
			if (bitmap != null)
			{
				pictureBoxDarknet.Image?.Dispose();
				pictureBoxDarknet.Image = bitmap;
			}
		}
	}

	private void trackBarCut_ValueChanged(object sender, EventArgs e)
	{
		lblThrCut.Text = trackBarCut.Value + "%";
	}

	private void trackBarRed_T_ValueChanged(object sender, EventArgs e)
	{
		lblThrRtape.Text = trackBarRed_T.Value + "%";
	}

	private void trackBarF_Body_ValueChanged(object sender, EventArgs e)
	{
		lblThrFbody.Text = trackBarF_Body.Value + "%";
	}

	protected override void Dispose(bool disposing)
	{
		if (disposing && components != null)
		{
			components.Dispose();
		}
		base.Dispose(disposing);
	}

	private void InitializeComponent()
	{
		this.components = new System.ComponentModel.Container();
		System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AI_Visual_Quality_Controller.Form1));
		this.splitContainer1 = new System.Windows.Forms.SplitContainer();
		this.splitContainer2 = new System.Windows.Forms.SplitContainer();
		this.camOutput = new System.Windows.Forms.Panel();
		this.pictureBoxDarknet = new System.Windows.Forms.PictureBox();
		this.txtConsoleOutput = new System.Windows.Forms.TextBox();
		this.groupBox3 = new System.Windows.Forms.GroupBox();
		this.label6 = new System.Windows.Forms.Label();
		this.lblF_Body = new System.Windows.Forms.Label();
		this.lblRed_T = new System.Windows.Forms.Label();
		this.lblCut = new System.Windows.Forms.Label();
		this.lblDressing = new System.Windows.Forms.Label();
		this.label5 = new System.Windows.Forms.Label();
		this.label7 = new System.Windows.Forms.Label();
		this.label4 = new System.Windows.Forms.Label();
		this.groupBox2 = new System.Windows.Forms.GroupBox();
		this.label9 = new System.Windows.Forms.Label();
		this.label11 = new System.Windows.Forms.Label();
		this.lblFPS = new System.Windows.Forms.Label();
		this.lblDPS = new System.Windows.Forms.Label();
		this.lblAvgFPS = new System.Windows.Forms.Label();
		this.label8 = new System.Windows.Forms.Label();
		this.groupBox1 = new System.Windows.Forms.GroupBox();
		this.lblThrCut = new System.Windows.Forms.Label();
		this.lblThrRtape = new System.Windows.Forms.Label();
		this.lblThrFbody = new System.Windows.Forms.Label();
		this.trackBarRed_T = new System.Windows.Forms.TrackBar();
		this.trackBarF_Body = new System.Windows.Forms.TrackBar();
		this.trackBarCut = new System.Windows.Forms.TrackBar();
		this.label1 = new System.Windows.Forms.Label();
		this.label2 = new System.Windows.Forms.Label();
		this.label3 = new System.Windows.Forms.Label();
		this.panelRedLight = new System.Windows.Forms.Panel();
		this.label10 = new System.Windows.Forms.Label();
		this.btnStartStop = new System.Windows.Forms.Button();
		this.timerUpdateDarknetWindow = new System.Windows.Forms.Timer(this.components);
		this.label12 = new System.Windows.Forms.Label();
		((System.ComponentModel.ISupportInitialize)this.splitContainer1).BeginInit();
		this.splitContainer1.Panel1.SuspendLayout();
		this.splitContainer1.Panel2.SuspendLayout();
		this.splitContainer1.SuspendLayout();
		((System.ComponentModel.ISupportInitialize)this.splitContainer2).BeginInit();
		this.splitContainer2.Panel1.SuspendLayout();
		this.splitContainer2.Panel2.SuspendLayout();
		this.splitContainer2.SuspendLayout();
		this.camOutput.SuspendLayout();
		((System.ComponentModel.ISupportInitialize)this.pictureBoxDarknet).BeginInit();
		this.groupBox3.SuspendLayout();
		this.groupBox2.SuspendLayout();
		this.groupBox1.SuspendLayout();
		((System.ComponentModel.ISupportInitialize)this.trackBarRed_T).BeginInit();
		((System.ComponentModel.ISupportInitialize)this.trackBarF_Body).BeginInit();
		((System.ComponentModel.ISupportInitialize)this.trackBarCut).BeginInit();
		this.panelRedLight.SuspendLayout();
		base.SuspendLayout();
		this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
		this.splitContainer1.FixedPanel = System.Windows.Forms.FixedPanel.Panel2;
		this.splitContainer1.IsSplitterFixed = true;
		this.splitContainer1.Location = new System.Drawing.Point(0, 0);
		this.splitContainer1.Name = "splitContainer1";
		this.splitContainer1.Panel1.Controls.Add(this.splitContainer2);
		this.splitContainer1.Panel2.BackColor = System.Drawing.SystemColors.Control;
		this.splitContainer1.Panel2.Controls.Add(this.groupBox3);
		this.splitContainer1.Panel2.Controls.Add(this.groupBox2);
		this.splitContainer1.Panel2.Controls.Add(this.groupBox1);
		this.splitContainer1.Panel2.Controls.Add(this.panelRedLight);
		this.splitContainer1.Panel2.Controls.Add(this.btnStartStop);
		this.splitContainer1.Size = new System.Drawing.Size(812, 627);
		this.splitContainer1.SplitterDistance = 639;
		this.splitContainer1.TabIndex = 0;
		this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
		this.splitContainer2.Location = new System.Drawing.Point(0, 0);
		this.splitContainer2.Name = "splitContainer2";
		this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
		this.splitContainer2.Panel1.Controls.Add(this.camOutput);
		this.splitContainer2.Panel2.Controls.Add(this.txtConsoleOutput);
		this.splitContainer2.Size = new System.Drawing.Size(639, 627);
		this.splitContainer2.SplitterDistance = 471;
		this.splitContainer2.TabIndex = 0;
		this.camOutput.Controls.Add(this.pictureBoxDarknet);
		this.camOutput.Dock = System.Windows.Forms.DockStyle.Fill;
		this.camOutput.Location = new System.Drawing.Point(0, 0);
		this.camOutput.Name = "camOutput";
		this.camOutput.Size = new System.Drawing.Size(639, 471);
		this.camOutput.TabIndex = 0;
		this.pictureBoxDarknet.BackColor = System.Drawing.Color.DarkSlateGray;
		this.pictureBoxDarknet.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
		this.pictureBoxDarknet.Dock = System.Windows.Forms.DockStyle.Fill;
		this.pictureBoxDarknet.Location = new System.Drawing.Point(0, 0);
		this.pictureBoxDarknet.Name = "pictureBoxDarknet";
		this.pictureBoxDarknet.Size = new System.Drawing.Size(639, 471);
		this.pictureBoxDarknet.TabIndex = 0;
		this.pictureBoxDarknet.TabStop = false;
		this.txtConsoleOutput.BackColor = System.Drawing.SystemColors.ScrollBar;
		this.txtConsoleOutput.Dock = System.Windows.Forms.DockStyle.Fill;
		this.txtConsoleOutput.Location = new System.Drawing.Point(0, 0);
		this.txtConsoleOutput.Multiline = true;
		this.txtConsoleOutput.Name = "txtConsoleOutput";
		this.txtConsoleOutput.Size = new System.Drawing.Size(639, 152);
		this.txtConsoleOutput.TabIndex = 0;
		this.groupBox3.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.groupBox3.Controls.Add(this.label6);
		this.groupBox3.Controls.Add(this.lblF_Body);
		this.groupBox3.Controls.Add(this.lblRed_T);
		this.groupBox3.Controls.Add(this.lblCut);
		this.groupBox3.Controls.Add(this.lblDressing);
		this.groupBox3.Controls.Add(this.label5);
		this.groupBox3.Controls.Add(this.label7);
		this.groupBox3.Controls.Add(this.label4);
		this.groupBox3.Location = new System.Drawing.Point(6, 491);
		this.groupBox3.Name = "groupBox3";
		this.groupBox3.Size = new System.Drawing.Size(154, 133);
		this.groupBox3.TabIndex = 25;
		this.groupBox3.TabStop = false;
		this.groupBox3.Text = "Coinfidence level in detection";
		this.label6.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label6.AutoSize = true;
		this.label6.Location = new System.Drawing.Point(12, 56);
		this.label6.Name = "label6";
		this.label6.Size = new System.Drawing.Size(68, 13);
		this.label6.TabIndex = 13;
		this.label6.Text = "Foreign body";
		this.lblF_Body.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblF_Body.AutoSize = true;
		this.lblF_Body.Location = new System.Drawing.Point(106, 56);
		this.lblF_Body.Name = "lblF_Body";
		this.lblF_Body.Size = new System.Drawing.Size(21, 13);
		this.lblF_Body.TabIndex = 10;
		this.lblF_Body.Text = "0%";
		this.lblRed_T.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblRed_T.AutoSize = true;
		this.lblRed_T.Location = new System.Drawing.Point(106, 84);
		this.lblRed_T.Name = "lblRed_T";
		this.lblRed_T.Size = new System.Drawing.Size(21, 13);
		this.lblRed_T.TabIndex = 11;
		this.lblRed_T.Text = "0%";
		this.lblCut.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblCut.AutoSize = true;
		this.lblCut.Location = new System.Drawing.Point(106, 111);
		this.lblCut.Name = "lblCut";
		this.lblCut.Size = new System.Drawing.Size(21, 13);
		this.lblCut.TabIndex = 12;
		this.lblCut.Text = "0%";
		this.lblDressing.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblDressing.AutoSize = true;
		this.lblDressing.Location = new System.Drawing.Point(106, 30);
		this.lblDressing.Name = "lblDressing";
		this.lblDressing.Size = new System.Drawing.Size(21, 13);
		this.lblDressing.TabIndex = 17;
		this.lblDressing.Text = "0%";
		this.label5.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label5.AutoSize = true;
		this.label5.Location = new System.Drawing.Point(12, 84);
		this.label5.Name = "label5";
		this.label5.Size = new System.Drawing.Size(51, 13);
		this.label5.TabIndex = 14;
		this.label5.Text = "Red tape";
		this.label7.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label7.AutoSize = true;
		this.label7.Location = new System.Drawing.Point(12, 30);
		this.label7.Name = "label7";
		this.label7.Size = new System.Drawing.Size(48, 13);
		this.label7.TabIndex = 16;
		this.label7.Text = "Dressing";
		this.label4.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label4.AutoSize = true;
		this.label4.Location = new System.Drawing.Point(12, 111);
		this.label4.Name = "label4";
		this.label4.Size = new System.Drawing.Size(23, 13);
		this.label4.TabIndex = 15;
		this.label4.Text = "Cut";
		this.groupBox2.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.groupBox2.Controls.Add(this.label9);
		this.groupBox2.Controls.Add(this.label11);
		this.groupBox2.Controls.Add(this.lblFPS);
		this.groupBox2.Controls.Add(this.lblDPS);
		this.groupBox2.Controls.Add(this.lblAvgFPS);
		this.groupBox2.Controls.Add(this.label8);
		this.groupBox2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25f, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, 0);
		this.groupBox2.Location = new System.Drawing.Point(8, 118);
		this.groupBox2.Name = "groupBox2";
		this.groupBox2.Size = new System.Drawing.Size(151, 106);
		this.groupBox2.TabIndex = 24;
		this.groupBox2.TabStop = false;
		this.groupBox2.Text = "Statistics";
		this.label9.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label9.AutoSize = true;
		this.label9.Location = new System.Drawing.Point(19, 53);
		this.label9.Name = "label9";
		this.label9.Size = new System.Drawing.Size(73, 13);
		this.label9.TabIndex = 20;
		this.label9.Text = "Average FPS:";
		this.label11.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label11.AutoSize = true;
		this.label11.Location = new System.Drawing.Point(9, 76);
		this.label11.Name = "label11";
		this.label11.Size = new System.Drawing.Size(92, 13);
		this.label11.TabIndex = 23;
		this.label11.Text = "Dressings/minute:";
		this.lblFPS.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblFPS.AutoSize = true;
		this.lblFPS.Location = new System.Drawing.Point(113, 28);
		this.lblFPS.Name = "lblFPS";
		this.lblFPS.Size = new System.Drawing.Size(13, 13);
		this.lblFPS.TabIndex = 1;
		this.lblFPS.Text = "0";
		this.lblDPS.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblDPS.AutoSize = true;
		this.lblDPS.Location = new System.Drawing.Point(113, 76);
		this.lblDPS.Name = "lblDPS";
		this.lblDPS.Size = new System.Drawing.Size(13, 13);
		this.lblDPS.TabIndex = 22;
		this.lblDPS.Text = "0";
		this.lblAvgFPS.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.lblAvgFPS.AutoSize = true;
		this.lblAvgFPS.Location = new System.Drawing.Point(113, 53);
		this.lblAvgFPS.Name = "lblAvgFPS";
		this.lblAvgFPS.Size = new System.Drawing.Size(13, 13);
		this.lblAvgFPS.TabIndex = 2;
		this.lblAvgFPS.Text = "0";
		this.label8.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label8.AutoSize = true;
		this.label8.Location = new System.Drawing.Point(31, 28);
		this.label8.Name = "label8";
		this.label8.Size = new System.Drawing.Size(30, 13);
		this.label8.TabIndex = 19;
		this.label8.Text = "FPS:";
		this.groupBox1.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.groupBox1.Controls.Add(this.label12);
		this.groupBox1.Controls.Add(this.lblThrCut);
		this.groupBox1.Controls.Add(this.lblThrRtape);
		this.groupBox1.Controls.Add(this.lblThrFbody);
		this.groupBox1.Controls.Add(this.trackBarRed_T);
		this.groupBox1.Controls.Add(this.trackBarF_Body);
		this.groupBox1.Controls.Add(this.trackBarCut);
		this.groupBox1.Controls.Add(this.label1);
		this.groupBox1.Controls.Add(this.label2);
		this.groupBox1.Controls.Add(this.label3);
		this.groupBox1.Location = new System.Drawing.Point(8, 230);
		this.groupBox1.Name = "groupBox1";
		this.groupBox1.Size = new System.Drawing.Size(154, 255);
		this.groupBox1.TabIndex = 21;
		this.groupBox1.TabStop = false;
		this.groupBox1.Text = "Threshold Adjustment";
		this.lblThrCut.AutoSize = true;
		this.lblThrCut.Location = new System.Drawing.Point(105, 194);
		this.lblThrCut.Name = "lblThrCut";
		this.lblThrCut.Size = new System.Drawing.Size(21, 13);
		this.lblThrCut.TabIndex = 20;
		this.lblThrCut.Text = "0%";
		this.lblThrRtape.AutoSize = true;
		this.lblThrRtape.Location = new System.Drawing.Point(104, 130);
		this.lblThrRtape.Name = "lblThrRtape";
		this.lblThrRtape.Size = new System.Drawing.Size(21, 13);
		this.lblThrRtape.TabIndex = 19;
		this.lblThrRtape.Text = "0%";
		this.lblThrFbody.AutoSize = true;
		this.lblThrFbody.Location = new System.Drawing.Point(105, 69);
		this.lblThrFbody.Name = "lblThrFbody";
		this.lblThrFbody.Size = new System.Drawing.Size(21, 13);
		this.lblThrFbody.TabIndex = 18;
		this.lblThrFbody.Text = "0%";
		this.trackBarRed_T.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.trackBarRed_T.Location = new System.Drawing.Point(-2, 145);
		this.trackBarRed_T.Maximum = 100;
		this.trackBarRed_T.Name = "trackBarRed_T";
		this.trackBarRed_T.Size = new System.Drawing.Size(155, 45);
		this.trackBarRed_T.TabIndex = 6;
		this.trackBarRed_T.Value = 80;
		this.trackBarRed_T.ValueChanged += new System.EventHandler(trackBarRed_T_ValueChanged);
		this.trackBarF_Body.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.trackBarF_Body.Location = new System.Drawing.Point(-2, 84);
		this.trackBarF_Body.Maximum = 100;
		this.trackBarF_Body.Name = "trackBarF_Body";
		this.trackBarF_Body.Size = new System.Drawing.Size(155, 45);
		this.trackBarF_Body.TabIndex = 4;
		this.trackBarF_Body.Value = 80;
		this.trackBarF_Body.ValueChanged += new System.EventHandler(trackBarF_Body_ValueChanged);
		this.trackBarCut.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.trackBarCut.Location = new System.Drawing.Point(-1, 209);
		this.trackBarCut.Maximum = 100;
		this.trackBarCut.Name = "trackBarCut";
		this.trackBarCut.Size = new System.Drawing.Size(155, 45);
		this.trackBarCut.TabIndex = 5;
		this.trackBarCut.Value = 80;
		this.trackBarCut.ValueChanged += new System.EventHandler(trackBarCut_ValueChanged);
		this.label1.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.label1.AutoSize = true;
		this.label1.Location = new System.Drawing.Point(9, 68);
		this.label1.Name = "label1";
		this.label1.Size = new System.Drawing.Size(68, 13);
		this.label1.TabIndex = 7;
		this.label1.Text = "Foreign body";
		this.label2.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.label2.AutoSize = true;
		this.label2.Location = new System.Drawing.Point(9, 129);
		this.label2.Name = "label2";
		this.label2.Size = new System.Drawing.Size(51, 13);
		this.label2.TabIndex = 8;
		this.label2.Text = "Red tape";
		this.label3.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.label3.AutoSize = true;
		this.label3.Location = new System.Drawing.Point(9, 193);
		this.label3.Name = "label3";
		this.label3.Size = new System.Drawing.Size(23, 13);
		this.label3.TabIndex = 9;
		this.label3.Text = "Cut";
		this.panelRedLight.BackColor = System.Drawing.Color.IndianRed;
		this.panelRedLight.Controls.Add(this.label10);
		this.panelRedLight.Dock = System.Windows.Forms.DockStyle.Top;
		this.panelRedLight.Location = new System.Drawing.Point(0, 0);
		this.panelRedLight.Name = "panelRedLight";
		this.panelRedLight.Size = new System.Drawing.Size(169, 31);
		this.panelRedLight.TabIndex = 18;
		this.panelRedLight.Visible = false;
		this.label10.Anchor = System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right;
		this.label10.AutoSize = true;
		this.label10.Location = new System.Drawing.Point(65, 9);
		this.label10.Name = "label10";
		this.label10.Size = new System.Drawing.Size(38, 13);
		this.label10.TabIndex = 22;
		this.label10.Text = "Reject";
		this.btnStartStop.Anchor = System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right;
		this.btnStartStop.BackColor = System.Drawing.Color.LimeGreen;
		this.btnStartStop.Font = new System.Drawing.Font("Microsoft Sans Serif", 12f, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, 0);
		this.btnStartStop.Location = new System.Drawing.Point(0, 40);
		this.btnStartStop.Name = "btnStartStop";
		this.btnStartStop.Size = new System.Drawing.Size(169, 72);
		this.btnStartStop.TabIndex = 0;
		this.btnStartStop.Text = "Start Detection";
		this.btnStartStop.UseVisualStyleBackColor = false;
		this.btnStartStop.Click += new System.EventHandler(btnStartStop_Click);
		this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25f, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, 0);
		this.label12.Location = new System.Drawing.Point(-2, 16);
		this.label12.Name = "label12";
		this.label12.Size = new System.Drawing.Size(150, 42);
		this.label12.TabIndex = 21;
		this.label12.Text = "Note: Increasing threshold lowers rejection rate";
		this.label12.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
		base.AutoScaleDimensions = new System.Drawing.SizeF(6f, 13f);
		base.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
		base.ClientSize = new System.Drawing.Size(812, 627);
		base.Controls.Add(this.splitContainer1);
		base.Icon = (System.Drawing.Icon)resources.GetObject("$this.Icon");
		base.Name = "Form1";
		this.Text = "AI Visual Quality Controller v0.52 by Dawid Olesko";
		this.splitContainer1.Panel1.ResumeLayout(false);
		this.splitContainer1.Panel2.ResumeLayout(false);
		((System.ComponentModel.ISupportInitialize)this.splitContainer1).EndInit();
		this.splitContainer1.ResumeLayout(false);
		this.splitContainer2.Panel1.ResumeLayout(false);
		this.splitContainer2.Panel2.ResumeLayout(false);
		this.splitContainer2.Panel2.PerformLayout();
		((System.ComponentModel.ISupportInitialize)this.splitContainer2).EndInit();
		this.splitContainer2.ResumeLayout(false);
		this.camOutput.ResumeLayout(false);
		((System.ComponentModel.ISupportInitialize)this.pictureBoxDarknet).EndInit();
		this.groupBox3.ResumeLayout(false);
		this.groupBox3.PerformLayout();
		this.groupBox2.ResumeLayout(false);
		this.groupBox2.PerformLayout();
		this.groupBox1.ResumeLayout(false);
		this.groupBox1.PerformLayout();
		((System.ComponentModel.ISupportInitialize)this.trackBarRed_T).EndInit();
		((System.ComponentModel.ISupportInitialize)this.trackBarF_Body).EndInit();
		((System.ComponentModel.ISupportInitialize)this.trackBarCut).EndInit();
		this.panelRedLight.ResumeLayout(false);
		this.panelRedLight.PerformLayout();
		base.ResumeLayout(false);
	}
}
