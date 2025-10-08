
#Write Flask code here 
from flask import Flask, request, jsonify, render_template
import time

app = Flask(__name__)

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Server is running"
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "online",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "message": "API is available"
    })

# ------------------ Error Handlers ------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Drishti AI Flask Server")
    print("=" * 60)
    
    print("\n📡 Available endpoints:")
    print("  🏠 /                           - Home page (index.html)")
    print("  ❤️  /health                    - Health check")
    print("  📊 /api/status                 - API status")
    
    print(f"\n🌐 Server: http://localhost:5000")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
