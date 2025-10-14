import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_labeling_server(port=8000):
    """
    Start a simple web server to serve the poultry labeling tool
    """
    # Get the current directory
    current_dir = Path.cwd()
    
    # Change to the directory with our files
    os.chdir(current_dir)
    
    # Set up the HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Poultry Labeling Server started!")
            print(f"📁 Serving directory: {current_dir}")
            print(f"🌐 Open your browser to: http://localhost:{port}")
            print(f"📄 Labeling tool available at: http://localhost:{port}/poultry_labeling_tool.html")
            print("Press Ctrl+C to stop the server")
            
            # Try to open the browser automatically
            try:
                webbrowser.open(f"http://localhost:{port}/poultry_labeling_tool.html")
            except:
                pass  # If browser opening fails, continue anyway
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    # Check if the HTML file exists
    if not os.path.exists("poultry_labeling_tool.html"):
        print("❌ Error: poultry_labeling_tool.html not found!")
        print("Please run csv_to_html_converter.py first to create the labeling tool")
        sys.exit(1)
    
    # Start the server on port 8000 (or first argument if provided)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_labeling_server(port)