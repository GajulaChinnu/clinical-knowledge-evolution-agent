import http.server
import socketserver
import os
from pathlib import Path

def start_server(port=8080):
    demo_dir = Path("data/demo").resolve()
    
    if not demo_dir.exists():
        print(f"Error: {demo_dir} does not exist.")
        return
        
    os.chdir(demo_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving HTTP on 0.0.0.0 port {port} (http://localhost:{port}/) ...")
        print(f"Demo file available at: http://localhost:{port}/demo_clinical_source.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    start_server()
