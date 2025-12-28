#!/usr/bin/env python3
"""
Main entry point for the Atari RL Server.
Starts the clean React-based dashboard with WebSocket streaming.
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description='Atari RL Training Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to bind to (default: 5001)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy streaming server')

    args = parser.parse_args()

    try:
        if args.legacy:
            # Legacy mode - use old streaming server
            print("🔄 Starting Legacy Streaming Server...")
            from streaming_server import IntegratedRLStreamingServer
            server = IntegratedRLStreamingServer()
            server.run(host=args.host, port=args.port, debug=args.debug)
        else:
            # New clean server
            print("=" * 60)
            print("🎮 Atari RL Training Dashboard")
            print("=" * 60)
            print(f"🌐 Starting server on http://{args.host}:{args.port}")
            print("📡 WebSocket streaming enabled")
            print("⚛️  React frontend")
            print("-" * 60)
            
            from server import run_server
            run_server(host=args.host, port=args.port, debug=args.debug)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
