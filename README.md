# HKL Fract.AI API

This is a Flask-based API for your YOLOv8-trained fracture detection model.

### Deployment (Render)
1. Push this folder to GitHub.
2. Go to [Render](https://render.com) → Create New Web Service.
3. Select your repo.
4. Environment: Python 3
5. Build command: `pip install -r requirements.txt`
6. Start command: `python app.py`
7. Deploy → Get your API endpoint: `https://fractai.onrender.com/predict`

### Testing locally
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000).

### Using the API
```python
import requests

url = "https://fractai.onrender.com/predict"
files = {'file': open('xray.jpg', 'rb')}
r = requests.post(url, files=files)
print(r.json())
```
