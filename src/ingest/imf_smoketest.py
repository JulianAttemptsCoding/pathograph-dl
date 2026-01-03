import os
import requests
import json
from datetime import datetime
import yaml

# Create necessary directories
os.makedirs("data/raw/imf_dots/_smoketest", exist_ok=True)
os.makedirs("data/raw/manifests", exist_ok=True)

# Define IMF SDMX endpoint (corrected)
IMF_BASE_URL = "https://sdmxcentral.imf.org/ws/public/sdmxapi/rest"
# Try the correct endpoint for DOTS (Direction of Trade Statistics)
DATAFLOW_ENDPOINT = f"{IMF_BASE_URL}/dataflow/all"


def test_imf_endpoint():
    """
    Test IMF SDMX endpoint connectivity
    """
    print("Testing IMF SDMX endpoint...")
    print(f"Endpoint: {DATAFLOW_ENDPOINT}")

    try:
        response = requests.get(DATAFLOW_ENDPOINT, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            # Try to parse the response to get dataflow count
            try:
                # Parse as XML first to get dataflow count
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)

                # Define namespace
                ns = {'mes': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
                      'str': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure'}

                # Count dataflows
                dataflows = root.findall('.//str:Dataflow', ns)
                dataflow_count = len(dataflows)

                print(f"✓ IMF endpoint reachable")
                print(f"✓ Found {dataflow_count} dataflows")

                # Save response headers and first 500 chars to log
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "url": DATAFLOW_ENDPOINT,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "response_preview": response.text[:500] if len(response.text) > 500 else response.text
                }

                log_path = "data/raw/imf_dots/_smoketest/smoketest_log.json"
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2)

                print(f"✓ Response log saved to: {log_path}")

                return True
            except Exception as e:
                print(f"Warning: Could not parse response: {e}")
                print("✓ IMF endpoint reachable but response format not recognized")

                # Still save basic response info
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "url": DATAFLOW_ENDPOINT,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "response_preview": response.text[:500] if len(response.text) > 500 else response.text
                }

                log_path = "data/raw/imf_dots/_smoketest/smoketest_log.json"
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2)

                print(f"✓ Response log saved to: {log_path}")

                return True
        else:
            print(f"✗ IMF endpoint returned status {response.status_code}")
            # Let's try another endpoint that might work for DOTS specifically
            dots_endpoint = f"{IMF_BASE_URL}/dataflow/IMF/DOT"
            print(f"Trying alternative endpoint: {dots_endpoint}")
            response2 = requests.get(dots_endpoint, timeout=30)
            print(f"Status Code for DOT endpoint: {response2.status_code}")
            if response2.status_code == 200:
                print("✓ Alternative IMF DOT endpoint is reachable")
                return True
            else:
                return False

    except requests.exceptions.ConnectionError:
        print("✗ Connection error - check internet connection and firewall settings")
        return False
    except requests.exceptions.Timeout:
        print("✗ Request timed out - server may be slow or unreachable")
        return False
    except Exception as e:
        print(f"✗ Error accessing IMF endpoint: {e}")
        return False


if __name__ == "__main__":
    success = test_imf_endpoint()
    if success:
        print("\n✓ IMF smoke test PASSED")
    else:
        print("\n✗ IMF smoke test FAILED")