from pathlib import Path
import requests
import socket

STRUCTURES_DIR = Path('data') / 'raw' / 'imf_dots' / '_structures'
host='dataservices.imf.org'
print('Resolving DNS for',host)
try:
    print(socket.gethostbyname(host))
except Exception as e:
    print('DNS resolution failed:',e)

print('\nTesting TCP connect to port 443')
import socket
s=socket.socket()
try:
    s.settimeout(10)
    s.connect((host,443))
    print('TCP connect: OK')
    s.close()
except Exception as e:
    print('TCP connect failed:',e)

print('\nHTTP GET /Dataflow')
try:
    r=requests.get('https://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow', timeout=20)
    print('HTTP status', r.status_code)
    print('Headers:', r.headers.get('content-type'))
    p=STRUCTURES_DIR / 'Dataflow_test.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text, encoding='utf-8')
    print('Saved sample to',p)
except Exception as e:
    print('HTTP GET failed:',e)
