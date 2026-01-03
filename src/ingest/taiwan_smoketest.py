import os
import requests
from datetime import datetime

# Create necessary directories
os.makedirs("data/raw/twn_mof/_smoketest", exist_ok=True)
os.makedirs("data/raw/manifests", exist_ok=True)

# Taiwan MOF/Customs data URL - replace with actual URL when known
# Example: "https://revenue-file.mof.gov.tw/TW/DownloadFile/TradeData_202301.csv"
# For now, using a placeholder - please update with actual Taiwan MOF URL
TAIWAN_MOF_URL = "https://revenue-file.mof.gov.tw/TW/DownloadFile/TradeData_202301.csv"  # Replace with actual URL


def test_taiwan_download():
    """
    Test Taiwan MOF data download
    """
    print("Testing Taiwan MOF data download...")
    print(f"URL: {TAIWAN_MOF_URL}")

    try:
        # Send GET request to download the file
        response = requests.get(TAIWAN_MOF_URL, timeout=60)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            # Determine file extension from response headers or URL
            content_type = response.headers.get('content-type', '')
            if 'application/zip' in content_type:
                file_ext = '.zip'
            elif 'text/csv' in content_type or 'application/csv' in content_type:
                file_ext = '.csv'
            elif 'application/vnd.ms-excel' in content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                file_ext = '.xlsx'
            else:
                # Try to extract from URL
                if TAIWAN_MOF_URL.endswith('.csv'):
                    file_ext = '.csv'
                elif TAIWAN_MOF_URL.endswith('.xlsx'):
                    file_ext = '.xlsx'
                elif TAIWAN_MOF_URL.endswith('.zip'):
                    file_ext = '.zip'
                else:
                    file_ext = '.dat'  # Default extension

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TWN_MOF_202301_smoketest_{timestamp}{file_ext}"
            filepath = f"data/raw/twn_mof/_smoketest/{filename}"

            # Write the content to file
            with open(filepath, 'wb') as f:
                f.write(response.content)

            file_size = len(response.content)
            print(f"✓ File downloaded successfully")
            print(f"✓ File saved to: {filepath}")
            print(f"✓ File size: {file_size} bytes")

            return True, filepath, file_size
        else:
            print(f"✗ Download failed with status {response.status_code}")
            print("Note: Taiwan MOF URL may have changed or require authentication")
            print("Please verify the correct download URL in Taiwan MOF/Customs website")
            return False, None, 0

    except requests.exceptions.ConnectionError:
        print("✗ Connection error - check internet connection and firewall settings")
        return False, None, 0
    except requests.exceptions.Timeout:
        print("✗ Request timed out - server may be slow or unreachable")
        return False, None, 0
    except Exception as e:
        print(f"✗ Error downloading Taiwan MOF data: {e}")
        print("Note: The Taiwan MOF URL might have changed or require special handling")
        return False, None, 0


if __name__ == "__main__":
    success, filepath, file_size = test_taiwan_download()
    if success:
        print(f"\n✓ Taiwan smoke test PASSED")
        print(f"✓ Downloaded file: {filepath} ({file_size} bytes)")
    else:
        print(f"\n✗ Taiwan smoke test FAILED")
        print("Please verify the Taiwan MOF URL in the script and try again")