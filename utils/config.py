import os
import socket
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL_FILE = os.path.join(PROJECT_ROOT, ".api_url")

def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_available_port(start_port: int = 8000, max_attempts: int = 20, host: str = "0.0.0.0") -> int:
    """
    Search for a free port starting from start_port up to start_port + max_attempts - 1.
    Raises RuntimeError if no free port is found within the threshold.
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port, host):
            if port != start_port:
                print(f"[INFO] Port {start_port} was occupied. Found available port: {port}")
            return port
        print(f"[INFO] Port {port} is busy, checking port {port + 1}...")
        
    error_msg = (
        f"[ERROR] No free port found in range {start_port} to {start_port + max_attempts - 1} "
        f"(checked {max_attempts} ports). Please free an existing port or specify PORT in your environment."
    )
    print(error_msg)
    raise RuntimeError(error_msg)

def save_api_url(port: int, host: str = "localhost"):
    """Persist active API URL for automatic discovery by UI/clients."""
    url = f"http://{host}:{port}"
    try:
        with open(API_URL_FILE, "w") as f:
            f.write(url)
    except Exception:
        pass

def get_api_url(default_host: str = "localhost", default_port: int = 8000) -> str:
    """
    Resolve active API URL in order of priority:
    1. API_URL environment variable
    2. .api_url local cache file
    3. Live probe for healthy backend (8000..8020)
    4. Default fallback http://{default_host}:{default_port}
    """
    if "API_URL" in os.environ and os.environ["API_URL"].strip():
        return os.environ["API_URL"].strip()

    if os.path.exists(API_URL_FILE):
        try:
            with open(API_URL_FILE, "r") as f:
                cached_url = f.read().strip()
                if cached_url:
                    # Quick probe to verify if cached URL is still alive
                    try:
                        r = requests.get(f"{cached_url}/health", timeout=0.3)
                        if r.status_code == 200:
                            return cached_url
                    except Exception:
                        pass
        except Exception:
            pass

    # Probe port range
    for port in range(default_port, default_port + 20):
        url = f"http://{default_host}:{port}"
        try:
            r = requests.get(f"{url}/health", timeout=0.1)
            if r.status_code == 200:
                return url
        except Exception:
            continue

    return f"http://{default_host}:{default_port}"
