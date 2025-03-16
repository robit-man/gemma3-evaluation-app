### A simple self contained frontend for evaluating gemma3 capabilities

- Function calling
- Webcam frame acquisition and inclusion in prompt
- Single command deployment using
```bash
python3 app.py
```

## example function to test function calling (chunk mode)
```bash
def get_ip_location(ip: str = None) -> dict:
    """
    Get the location details of your public IP address using the ipconfig.io service.
    
    The function accepts an optional 'ip' parameter, but it is ignored. It always retrieves
    your public IP automatically via ipconfig.io.
    
    Returns:
      A dictionary containing location information (such as city, region, country, etc.).
      In case of an error, returns a dict with an "error" key.
    """
    import requests
    try:
        response = requests.get("https://ipconfig.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}
```
