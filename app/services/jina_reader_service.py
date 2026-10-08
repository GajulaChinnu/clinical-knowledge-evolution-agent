import json
import logging
import os
import socket
import urllib.error
import urllib.request
from typing import Dict, Tuple

logger = logging.getLogger("ckea.services.jina_reader")

class JinaReaderError(Exception):
    pass

class JinaReaderService:
    """Dedicated retrieval service for generic webpage URLs using Jina Reader."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        self.base_url = base_url or os.environ.get("JINA_READER_BASE_URL", "https://r.jina.ai")

    def retrieve_webpage(self, url: str, timeout_seconds: float = 20.0) -> Tuple[str, str, str]:
        """Retrieve web page content via Jina Reader.
        
        Returns:
            Tuple of (markdown_content, page_title, resolved_url)
        """
        if not self.api_key:
            raise JinaReaderError("JINA_API_KEY is not configured.")
        
        target_url = f"{self.base_url.rstrip('/')}/{url}"
        
        req = urllib.request.Request(
            target_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "X-Return-Format": "markdown"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                status_code = getattr(response, "status", 200)
                if status_code >= 400:
                    raise JinaReaderError(f"Jina API returned {status_code}")
                    
                content_bytes = response.read()
                
                try:
                    data = json.loads(content_bytes)
                except json.JSONDecodeError:
                    raise JinaReaderError("Jina API returned invalid JSON")
                    
                # Handle error responses from Jina
                if data.get("code") != 200:
                    raise JinaReaderError(f"Jina API error {data.get('code')}: {data.get('message', 'Unknown error')}")
                
                res_data = data.get("data", {})
                if not isinstance(res_data, dict):
                    res_data = {}
                    
                content = res_data.get("content", "")
                title = res_data.get("title", "")
                resolved_url = res_data.get("url", url)
                
                return content, title, resolved_url
                
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise JinaReaderError(f"Jina API Authentication error ({e.code}). Check JINA_API_KEY.")
            elif e.code == 402:
                raise JinaReaderError("Jina API Quota exceeded (402).")
            elif e.code == 429:
                raise JinaReaderError("Jina API Rate limited (429).")
            elif e.code >= 500:
                raise JinaReaderError(f"Jina API Server error ({e.code}).")
            else:
                raise JinaReaderError(f"HTTP {e.code} error fetching from Jina: {e.reason}")
                
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                raise JinaReaderError(f"Request to Jina API timed out after {timeout_seconds}s")
            raise JinaReaderError(f"Network error communicating with Jina API: {e.reason}")
            
        except (socket.timeout, TimeoutError):
            raise JinaReaderError(f"Request to Jina API timed out after {timeout_seconds}s")
            
        except JinaReaderError:
            raise
            
        except Exception as e:
            raise JinaReaderError(f"Unexpected error with Jina Reader: {e}")
