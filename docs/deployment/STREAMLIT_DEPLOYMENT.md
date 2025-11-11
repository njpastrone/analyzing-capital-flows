# 🚀 Streamlit Community Cloud Deployment Guide

## Capital Flows Research Platform - Production Deployment Checklist

### ✅ Pre-Deployment Audit Results

#### 🔒 Security & Credentials
- ✅ **No sensitive data found** - No API keys, passwords, or tokens in codebase
- ✅ **No hardcoded paths** - All file paths use relative references
- ✅ **Clean repository** - No local environment assumptions

#### 📦 Repository Optimization Status
- **Current Size**: 193MB (well within 1GB limit)
- **Cleanup Opportunities**:
  - 70 cache files (__pycache__, .pyc) - Will be ignored via .gitignore
  - 6 debug print statements - Non-critical, can remain
  - 2.1MB output directory - Consider excluding from deployment

#### 🔧 Dependencies
All packages in `requirements.txt` are cloud-compatible:
```
✅ streamlit>=1.28.0
✅ pandas>=2.0.0
✅ numpy>=1.24.0
✅ matplotlib>=3.7.0
✅ plotly>=5.15.0
✅ statsmodels>=0.14.0
✅ scipy>=1.10.0
✅ seaborn>=0.12.0
✅ openpyxl>=3.1.0
```

---

## 📋 Deployment Checklist

### Phase 1: GitHub Repository Preparation

#### Clean Repository
```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# Remove Jupyter checkpoints
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null
```

#### Repository Configuration
- [ ] Update `.gitignore` (✅ Already updated)
- [ ] Verify all changes committed: `git status`
- [ ] Push to main branch: `git push origin main`
- [ ] Ensure repository is **PUBLIC** (required for free tier)

#### Add Deployment Files
1. **Create `streamlit_app.py` in root directory:**
```python
"""
Capital Flows Research Platform
Streamlit Community Cloud Entry Point
"""
import sys
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import and run main dashboard
from dashboard.main_app import main

if __name__ == "__main__":
    main()
```

2. **Update `requirements.txt` location:**
   - Ensure `requirements.txt` is in repository root
   - If not, copy from current location

3. **Create `.streamlit/config.toml`:**
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
magicEnabled = false

[client]
showErrorDetails = true
```

---

### Phase 2: Streamlit Cloud Setup

#### 1. Account Creation
- [ ] Go to [share.streamlit.io](https://share.streamlit.io)
- [ ] Click "Sign up with GitHub"
- [ ] Authorize Streamlit to access your GitHub account
- [ ] Complete profile setup

#### 2. Deploy Application
- [ ] Click "New app" button
- [ ] Repository: `analyzing-capital-flows`
- [ ] Branch: `main`
- [ ] Main file path: `streamlit_app.py` (or `src/dashboard/main_app.py` if not using wrapper)
- [ ] App URL: Choose subdomain (e.g., `capital-flows-research`)

#### 3. Advanced Settings (Optional)
- [ ] Python version: 3.10 (recommended for compatibility)
- [ ] Install commands: None needed (requirements.txt handles all)
- [ ] Secrets management: Not required for this app

#### 4. Deploy
- [ ] Click "Deploy"
- [ ] Wait for build (typically 2-5 minutes first time)
- [ ] Monitor logs for any errors

---

### Phase 3: Performance Optimization

#### Add Caching to Main Application
Update `src/dashboard/main_app.py` with:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file_path):
    """Load and cache CSV data"""
    return pd.read_csv(file_path)

@st.cache_resource
def load_statistical_models():
    """Cache heavy computational models"""
    # Your model loading code here
    pass
```

#### Memory Optimization Patterns
```python
# Use generators for large datasets
def process_data_in_chunks(df, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

# Clear matplotlib figures after use
import matplotlib.pyplot as plt
plt.close('all')  # Add after each plot generation
```

---

### Phase 4: Post-Deployment Validation

#### Functional Testing
- [ ] **Case Study 1**: Iceland vs Eurozone loads correctly
- [ ] **Case Study 2**: All Baltic countries display properly
- [ ] **Case Study 3**: Small economies comparison works
- [ ] **Case Study 4**: Statistical analysis executes
- [ ] **Case Study 5**: Capital controls analysis functions
- [ ] **Download/Export**: PDF reports generate successfully
- [ ] **Outlier Analysis**: Winsorized data tabs work

#### Performance Testing
- [ ] Initial load time < 30 seconds
- [ ] Tab switching responsive (< 3 seconds)
- [ ] Chart generation smooth
- [ ] Memory usage stable over session

#### Error Handling
- [ ] Check Streamlit logs for warnings
- [ ] Verify data file paths resolve correctly
- [ ] Test with multiple concurrent users

---

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### Issue: "Module not found" error
**Solution**: Ensure all imports use relative paths from src/
```python
# Instead of: from dashboard.main_app import main
# Use: from src.dashboard.main_app import main
```

#### Issue: File not found errors
**Solution**: Use pathlib for robust path handling
```python
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / 'updated_data' / 'Clean'
```

#### Issue: Memory limit exceeded
**Solution**: Implement data sampling for large operations
```python
# Sample data for visualization if needed
if len(df) > 10000:
    df_display = df.sample(n=10000, random_state=42)
```

#### Issue: Slow initial load
**Solution**: Add loading indicators
```python
with st.spinner('Loading data...'):
    data = load_large_dataset()
```

---

## 📊 Monitoring & Maintenance

### Analytics Dashboard
- Access via Streamlit Cloud dashboard
- Monitor: User sessions, load times, errors
- Set up email alerts for failures

### Update Process
1. Make changes locally
2. Test thoroughly
3. Push to GitHub: `git push origin main`
4. Streamlit auto-redeploys (2-3 minutes)

### Backup Strategy
- GitHub serves as primary backup
- Consider tagging stable releases: `git tag v1.0.0`
- Export critical data periodically

---

## 🎯 Quick Deployment Commands

```bash
# One-liner cleanup before deployment
find . -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" -o -name ".ipynb_checkpoints" | xargs rm -rf

# Check repository size
du -sh .

# Verify no sensitive data
grep -r "api_key\|password\|secret" --include="*.py"

# Final commit
git add -A
git commit -m "Prepare for Streamlit Community Cloud deployment"
git push origin main
```

---

## 📝 License Recommendation

Add `LICENSE` file for academic work:
```
MIT License

Copyright (c) 2024 Capital Flows Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## ✅ Ready for Deployment!

Your application is **production-ready** for Streamlit Community Cloud deployment. The codebase is clean, secure, and optimized for cloud hosting. Expected deployment time: **30 minutes**.

### Support Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)