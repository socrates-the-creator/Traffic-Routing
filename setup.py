"""
Setup script for Mumbai Navigation System.
This script helps with initial setup and configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "logs",
        "data/backup"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    print("✅ Directories created successfully")

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Mumbai Navigation System Environment Variables\n")
            f.write("# Add your TomTom API key here\n")
            f.write("TOMTOM_API_KEY=your_tomtom_api_key_here\n")
            f.write("\n# Flask configuration\n")
            f.write("FLASK_ENV=development\n")
            f.write("FLASK_DEBUG=True\n")
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your TomTom API key")
    else:
        print("✅ .env file already exists")

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    return True

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Data files
data/raw/*.csv
data/raw/*.geojson
data/processed/*.pkl
data/models/*.h5
data/models/*.pkl
data/backup/

# Logs
logs/*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints/

# Model checkpoints
*.ckpt
*.pth
"""
    
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        print("📝 Creating .gitignore file...")
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ .gitignore file created")
    else:
        print("✅ .gitignore file already exists")

def test_installation():
    """Test if the installation is working."""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import folium
        print("✅ Folium imported successfully")
    except ImportError as e:
        print(f"❌ Folium import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    print("✅ All core dependencies imported successfully")
    return True

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    # Create sample traffic data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    sample_traffic = pd.DataFrame({
        'segment_id': [f'segment_{i}' for i in range(10)],
        'latitude': np.random.uniform(18.9, 19.3, 10),
        'longitude': np.random.uniform(72.8, 73.2, 10),
        'current_speed': np.random.uniform(20, 60, 10),
        'free_flow_speed': np.random.uniform(40, 80, 10),
        'jam_factor': np.random.uniform(0, 0.8, 10),
        'traffic_level': np.random.choice(['FLOWING', 'SLOW', 'CONGESTED'], 10),
        'confidence': np.random.uniform(0.7, 1.0, 10),
        'timestamp': pd.Timestamp.now().isoformat()
    })
    
    sample_traffic.to_csv('data/raw/sample_traffic.csv', index=False)
    print("✅ Sample traffic data created")
    
    # Create sample historical data
    sample_historical = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'segment_id': np.random.choice([f'segment_{i}' for i in range(10)], 100),
        'speed': np.random.uniform(20, 60, 100),
        'volume': np.random.poisson(100, 100),
        'day_of_week': np.random.randint(0, 7, 100),
        'hour': np.random.randint(0, 24, 100),
        'is_weekend': np.random.choice([True, False], 100)
    })
    
    sample_historical.to_csv('data/raw/sample_historical.csv', index=False)
    print("✅ Sample historical data created")

def main():
    """Main setup function."""
    print("🚀 Setting up Mumbai Navigation System...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Create .gitignore
    create_gitignore()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your TomTom API key")
    print("2. Run: python src/ingest/osm_overpass.py")
    print("3. Run: python src/ingest/tomtom_fetch.py")
    print("4. Run: python src/preprocess/build_graph.py")
    print("5. Run: python src/models/train.py")
    print("6. Run: python app/app.py")
    print("7. Open http://localhost:5000 in your browser")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
