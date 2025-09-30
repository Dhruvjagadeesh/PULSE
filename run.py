#!/usr/bin/env python3
"""
PULSE - Emotional Wellness Application
Run this script to start the Flask server
"""

from app import app

if __name__ == '__main__':
    print("🚀 Starting PULSE - Emotional Wellness Application")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("❌ Press Ctrl+C to stop the server")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
