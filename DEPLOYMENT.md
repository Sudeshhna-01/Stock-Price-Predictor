# Streamlit Deployment Guide

## Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:8501`

## Streamlit Cloud Deployment

### Method 1: Direct GitHub Deployment

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/stock-forecasting-app.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path to `app.py`
   - Click Deploy

### Method 2: Manual Deployment

1. **Create a `packages.txt` file** for system dependencies:
   ```
   # packages.txt
   ```

2. **Update `requirements.txt`** for deployment:
   ```
   streamlit
   yfinance
   pandas
   numpy
   matplotlib
   seaborn
   statsmodels
   scikit-learn
   torch
   pmdarima
   plotly
   ```

3. **Create `.streamlit/config.toml`:**
   ```toml
   [global]
   developmentMode = false
   dataFrameSerialization = "legacy"

   [logger]
   level = "info"

   [client]
   showSidebarNavigation = true
   showErrorDetails = false
   ```

## Heroku Deployment

1. **Create `Procfile`:**
   ```
   web: streamlit run app.py --server.port $PORT --server.headless true
   ```

2. **Create `setup.sh`:**
   ```bash
   #!/bin/bash
   mkdir -p ~/.streamlit/
   echo "[global]
   developmentMode = false
   dataFrameSerialization = "legacy"
   " > ~/.streamlit/config.toml
   echo "[logger]
   level = "info"
   " >> ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Docker Deployment

1. **Create `Dockerfile`:**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
   ```

2. **Create `docker-compose.yml`:**
   ```yaml
   version: '3.8'
   services:
     streamlit:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - .:/app
   ```

3. **Run with Docker:**
   ```bash
   docker-compose up --build
   ```

## Production Optimizations

1. **Caching:** Add `@st.cache_data` decorators for expensive operations
2. **Session State:** Use `st.session_state` for maintaining state
3. **Error Handling:** Add try-except blocks for robustness
4. **Loading States:** Use `st.spinner()` for long-running operations
5. **Resource Limits:** Set appropriate memory limits for cloud deployments

## Environment Variables

Create a `.env` file for sensitive configurations:
```
API_KEY=your_api_key_here
DATABASE_URL=your_database_url
```

## Security Considerations

1. **Input Validation:** Validate all user inputs
2. **Rate Limiting:** Implement rate limiting for API calls
3. **Data Sanitization:** Clean and validate data before processing
4. **Error Messages:** Don't expose sensitive error information
5. **HTTPS:** Always use HTTPS in production

## Monitoring

- **Streamlit Cloud:** Built-in analytics and monitoring
- **Custom Logging:** Add logging for user interactions
- **Performance Monitoring:** Track response times and resource usage

## Troubleshooting

### Common Issues:

1. **Memory Errors:** Reduce batch sizes or use smaller models
2. **Timeout Errors:** Add timeout handling for long operations
3. **Import Errors:** Ensure all dependencies are in requirements.txt
4. **Port Issues:** Check if port 8501 is available

### Performance Tips:

1. **Use caching** for repeated computations
2. **Optimize data loading** with efficient pandas operations
3. **Reduce model complexity** for faster inference
4. **Use async operations** for non-blocking UI updates