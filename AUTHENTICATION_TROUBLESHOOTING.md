# Authentication Troubleshooting Guide

## Common Authentication Issues in Cloud Run Deployment

### 1. Session Storage Issues

**Problem**: Sessions are lost when Cloud Run instances restart or scale.

**Symptoms**:
- User gets logged out unexpectedly
- Login page appears after successful authentication
- Authentication state doesn't persist across page refreshes

**Solutions**:
- ✅ **Implemented**: Session expiration (24 hours)
- ✅ **Implemented**: Automatic session cleanup
- ✅ **Implemented**: Session validation on app load
- ✅ **Implemented**: Debug logging for authentication flow

### 2. Callback Dependencies

**Problem**: Authentication callbacks may not fire in the correct order.

**Symptoms**:
- Login page shows even after successful login
- Authentication state doesn't update properly
- Callback errors in browser console

**Solutions**:
- ✅ **Implemented**: `prevent_initial_call=False` for session validation
- ✅ **Implemented**: `allow_duplicate=True` for auth state updates
- ✅ **Implemented**: Debug logging for all auth callbacks

### 3. Environment Variables

**Problem**: Missing or incorrect environment variables in Cloud Run.

**Symptoms**:
- App fails to start
- BigQuery connection errors
- Secret Manager access denied

**Solutions**:
- ✅ **Implemented**: Proper environment variable setup in deploy.sh
- ✅ **Implemented**: Cloud Run environment detection (`K_SERVICE`)
- ✅ **Implemented**: Fallback mechanisms for missing variables

### 4. Network and CORS Issues

**Problem**: Cloud Run may have different network behavior than local development.

**Symptoms**:
- Callbacks fail to execute
- Authentication requests timeout
- CORS errors in browser

**Solutions**:
- ✅ **Implemented**: Health check endpoint (`/health`)
- ✅ **Implemented**: Proper host binding (`0.0.0.0`)
- ✅ **Implemented**: Cloud Run specific configuration

## Debugging Steps

### 1. Check Cloud Run Logs
```bash
gcloud run services logs read supply-shed-visualizer --region europe-west2 --limit 100
```

Look for:
- `🔐 Auth state:` - Shows authentication state changes
- `🔍 Validating session:` - Shows session validation attempts
- `✅ Login successful` - Confirms successful logins
- `❌ Session invalid or expired` - Shows session issues

### 2. Test Health Endpoint
```bash
curl https://your-service-url/health
```

Should return:
```json
{"status": "healthy", "service": "supply-shed-visualizer"}
```

### 3. Check Environment Variables
```bash
gcloud run services describe supply-shed-visualizer --region europe-west2
```

Verify these are set:
- `BIGQUERY_PROJECT_ID=epoch-geospatial-dev`
- `GOOGLE_CLOUD_PROJECT=epoch-geospatial-dev`
- `DEBUG=False`
- `K_SERVICE=supply-shed-visualizer`

### 4. Test Authentication Flow

1. **Open browser developer tools**
2. **Navigate to the app**
3. **Check console for errors**
4. **Try logging in with**: `william@epoch.blue` / `ssi123`
5. **Watch for authentication debug messages**

## Authentication Flow

### 1. App Startup
```
1. App loads with auth-state = {'authenticated': False}
2. validate_session_callback runs (no session_id initially)
3. update_main_content shows login page
```

### 2. Login Process
```
1. User enters credentials and clicks login
2. handle_login callback validates credentials
3. If valid: creates session and updates auth-state
4. update_main_content shows main layout
```

### 3. Session Validation
```
1. validate_session_callback runs periodically
2. Checks if session exists and is not expired
3. Updates auth-state accordingly
4. cleanup_expired_sessions removes old sessions
```

## Common Fixes

### Fix 1: Clear Browser Storage
```javascript
// In browser console
localStorage.clear();
sessionStorage.clear();
location.reload();
```

### Fix 2: Check Service Account Permissions
```bash
# Verify service account has required roles
gcloud projects get-iam-policy epoch-geospatial-dev \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:709579113971-compute@developer.gserviceaccount.com"
```

### Fix 3: Redeploy with Debug Mode
```bash
# Temporarily enable debug mode
gcloud run services update supply-shed-visualizer \
    --region europe-west2 \
    --set-env-vars "DEBUG=True"
```

### Fix 4: Check BigQuery Connection
```bash
# Test BigQuery access
gcloud auth application-default login
bq query --use_legacy_sql=false "SELECT 1 as test"
```

## Production Recommendations

### 1. Use Redis for Session Storage
For production, replace in-memory session storage with Redis:

```python
import redis
import json

redis_client = redis.Redis(host='your-redis-host', port=6379, db=0)

def create_session(username):
    session_id = secrets.token_urlsafe(32)
    session_data = {
        'authenticated': True,
        'user': {'email': username, 'name': username.split('@')[0].title()},
        'created_at': time.time(),
        'expires_at': time.time() + SESSION_TIMEOUT
    }
    redis_client.setex(session_id, SESSION_TIMEOUT, json.dumps(session_data))
    return session_id
```

### 2. Add Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app.server,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.server.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login_endpoint():
    # Login logic here
    pass
```

### 3. Add HTTPS Enforcement
```python
@app.server.before_request
def force_https():
    if not request.is_secure and os.getenv('K_SERVICE'):
        return redirect(request.url.replace('http://', 'https://'))
```

## Contact Support

If issues persist:
1. Check Cloud Run logs for specific error messages
2. Verify all environment variables are set correctly
3. Test the health endpoint
4. Check service account permissions
5. Review the authentication flow debug messages

The authentication system is designed to be robust and provide detailed logging for troubleshooting.
