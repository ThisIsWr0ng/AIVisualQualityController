using System;
using System.Windows.Forms;

namespace AI_Visual_Quality_Controller;

internal static class Program
{
	[STAThread]
	private static void Main()
	{
		Application.EnableVisualStyles();
		Application.SetCompatibleTextRenderingDefault(defaultValue: false);
		Application.Run(new Form1());
	}
}
