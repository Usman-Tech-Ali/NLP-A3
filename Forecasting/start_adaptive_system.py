"""
Start Adaptive Learning System

Quick start script that:
1. Checks MongoDB connection
2. Initializes the system if needed
3. Starts the Flask application
"""

import subprocess
import sys
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def check_mongodb():
    """Check if MongoDB is running"""
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print("✅ MongoDB is running")
        return True
    except ConnectionFailure:
        print("❌ MongoDB is not running!")
        print("\n💡 Please start MongoDB:")
        print("   - Windows: Run 'mongod' in a separate terminal")
        print("   - Mac: Run 'brew services start mongodb-community'")
        print("   - Linux: Run 'sudo systemctl start mongod'")
        return False


def check_initialization():
    """Check if system has been initialized"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['stock_forecasting']
        
        # Check if we have any performance data
        count = db['performance_history'].count_documents({})
        
        if count > 0:
            print(f"✅ System already initialized ({count} predictions logged)")
            return True
        else:
            print("⚠️  System not initialized yet")
            return False
    except Exception as e:
        print(f"⚠️  Could not check initialization: {e}")
        return False


def initialize_system():
    """Run initialization script"""
    print("\n🚀 Initializing adaptive learning system...")
    print("This will take a few minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'backend/initialize_adaptive_system.py'],
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Initialization failed: {e}")
        return False


def start_flask():
    """Start Flask application"""
    print("\n🌐 Starting Flask application...")
    print("=" * 60)
    print("📍 Main App: http://localhost:5000")
    print("🤖 Monitor:  http://localhost:5000/monitor")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            [sys.executable, 'backend/app.py'],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Flask failed to start: {e}")


def main():
    """Main startup sequence"""
    print("=" * 60)
    print("🤖 ADAPTIVE LEARNING SYSTEM - STARTUP")
    print("=" * 60)
    print()
    
    # Step 1: Check MongoDB
    if not check_mongodb():
        sys.exit(1)
    
    # Step 2: Check initialization
    is_initialized = check_initialization()
    
    if not is_initialized:
        response = input("\n❓ Initialize system now? (y/n): ").lower()
        if response == 'y':
            if not initialize_system():
                print("\n❌ Initialization failed. Please check the errors above.")
                sys.exit(1)
        else:
            print("\n⚠️  Starting without initialization...")
            print("   You can initialize later by running:")
            print("   python backend/initialize_adaptive_system.py")
    
    # Step 3: Start Flask
    start_flask()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
