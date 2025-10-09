"""
Quick startup script for Mumbai Navigation System.
This script provides easy access to all system functions.
"""

import os
import sys
import subprocess
import argparse

def run_setup():
    """Run the setup script."""
    print("🔧 Running setup...")
    subprocess.run([sys.executable, "setup.py"])

def run_training():
    """Run the training script."""
    print("🤖 Running GNN training...")
    subprocess.run([sys.executable, "training.py"])

def run_main():
    """Run the main application."""
    print("🚀 Running main application...")
    subprocess.run([sys.executable, "main.py"])

def run_demo():
    """Run the demo script."""
    print("🎯 Running demo...")
    subprocess.run([sys.executable, "demo.py"])

def run_quick_training():
    """Run quick training for testing."""
    print("⚡ Running quick training...")
    subprocess.run([sys.executable, "training.py", "--quick"])

def run_data_collection():
    """Run data collection only."""
    print("📊 Running data collection...")
    subprocess.run([sys.executable, "training.py", "--data-only"])

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Mumbai Navigation System - Quick Runner')
    parser.add_argument('command', nargs='?', choices=[
        'setup', 'train', 'main', 'demo', 'quick-train', 'data', 'help'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    print("🎯 Mumbai Navigation System - Quick Runner")
    print("=" * 50)
    
    if not args.command:
        # Interactive mode
        while True:
            print("\n📋 Available commands:")
            print("1. setup      - Run initial setup")
            print("2. train      - Train GNN model")
            print("3. main       - Run main application")
            print("4. demo       - Run demo")
            print("5. quick-train - Quick training for testing")
            print("6. data       - Collect data only")
            print("7. help       - Show help")
            print("8. exit       - Exit")
            
            choice = input("\nEnter command (1-8): ").strip()
            
            if choice == '1' or choice == 'setup':
                run_setup()
            elif choice == '2' or choice == 'train':
                run_training()
            elif choice == '3' or choice == 'main':
                run_main()
            elif choice == '4' or choice == 'demo':
                run_demo()
            elif choice == '5' or choice == 'quick-train':
                run_quick_training()
            elif choice == '6' or choice == 'data':
                run_data_collection()
            elif choice == '7' or choice == 'help':
                show_help()
            elif choice == '8' or choice == 'exit':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")
    else:
        # Command line mode
        if args.command == 'setup':
            run_setup()
        elif args.command == 'train':
            run_training()
        elif args.command == 'main':
            run_main()
        elif args.command == 'demo':
            run_demo()
        elif args.command == 'quick-train':
            run_quick_training()
        elif args.command == 'data':
            run_data_collection()
        elif args.command == 'help':
            show_help()

def show_help():
    """Show help information."""
    print("\n📚 Mumbai Navigation System Help")
    print("=" * 50)
    print("\n🚀 Quick Start:")
    print("1. python run.py setup     # Initial setup")
    print("2. python run.py data      # Collect data")
    print("3. python run.py train     # Train GNN model")
    print("4. python run.py main      # Run web application")
    
    print("\n📋 Available Commands:")
    print("• setup      - Run initial setup and install dependencies")
    print("• train      - Train the GNN model with full pipeline")
    print("• main       - Run the main application with web interface")
    print("• demo       - Run a demonstration of system features")
    print("• quick-train - Quick training for testing (3 days data, 10 epochs)")
    print("• data       - Collect training data only")
    print("• help       - Show this help message")
    
    print("\n🔧 Manual Commands:")
    print("• python setup.py          - Setup system")
    print("• python training.py       - Train GNN model")
    print("• python main.py           - Run main application")
    print("• python demo.py           - Run demo")
    
    print("\n📁 Project Structure:")
    print("• src/                     - Source code")
    print("• app/                     - Web application")
    print("• data/                    - Data files")
    print("• logs/                    - Log files")
    print("• results/                 - Training results")
    
    print("\n🌐 Web Interface:")
    print("• After running 'main', open http://localhost:5000")
    print("• Interactive map with route planning")
    print("• Real-time traffic visualization")
    print("• Algorithm comparison")
    
    print("\n🤖 GNN Model:")
    print("• Graph Neural Network for traffic prediction")
    print("• Trained on historical traffic data")
    print("• Provides intelligent route recommendations")
    print("• Compares with A* and Dijkstra algorithms")

if __name__ == "__main__":
    main()
