#!/usr/bin/env python3
"""
Simple script to run the AgriGraph AI web application.
"""

from agrigraph_ai.web_app import run_web_app

if __name__ == '__main__':
    run_web_app(host='127.0.0.1', port=5000, debug=True)







