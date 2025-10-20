# Fix: Login State Persistence After Page Refresh

## Problem
After refreshing the page at `localhost:5173`, the application would lose its login state and redirect users to the login page, even though valid authentication tokens existed in localStorage.

## Root Cause
The issue was caused by a race condition during MSAL initialization:

1. **MSAL stores tokens in localStorage** (`cacheLocation: 'localStorage'` in config)
2. **React renders immediately** when the app loads
3. **MSAL hooks need time** to read from localStorage and rehydrate the authentication state
4. **Premature check** - The app was checking `isAuthenticated` before MSAL finished loading accounts from cache
5. **Result** - Users briefly appeared as "not authenticated" during initialization, triggering a redirect to the login page

## Solution Implemented

### Changes to `App.jsx`

1. **Added initialization state tracking:**
   ```jsx
   const [isInitializing, setIsInitializing] = useState(true);
   ```

2. **Added MSAL `inProgress` status:**
   ```jsx
   const { instance, accounts, inProgress } = useMsal();
   ```

3. **Added initialization delay:**
   ```jsx
   useEffect(() => {
     // Give MSAL a moment to load accounts from cache
     const timer = setTimeout(() => {
       setIsInitializing(false);
     }, 500);
     
     return () => clearTimeout(timer);
   }, []);
   ```

4. **Added loading screen during initialization:**
   ```jsx
   // Show loading state while MSAL is initializing to prevent flash of login page
   if (isInitializing || inProgress === 'startup' || inProgress === 'handleRedirect') {
     return (
       <div className="container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
         <div style={{ textAlign: 'center' }}>
           <div style={{ fontSize: '24px', marginBottom: '10px' }}>Loading...</div>
           <div style={{ fontSize: '14px', color: '#666' }}>Checking authentication status</div>
         </div>
       </div>
     );
   }
   ```

## How It Works Now

1. **Page loads** → App component mounts with `isInitializing = true`
2. **Loading screen displays** → Prevents premature authentication check
3. **MSAL initialization completes** → Reads tokens from localStorage (happens in `main.jsx`)
4. **Accounts rehydrate** → MSAL loads cached accounts
5. **After 500ms timeout** → `isInitializing` becomes `false`
6. **Authentication check runs** → Now with properly loaded accounts
7. **Correct page renders** → Either main app (if authenticated) or login page

## Key Benefits

✅ **No more login redirects** after page refresh  
✅ **Smooth user experience** with loading indicator  
✅ **Proper initialization** - waits for MSAL to be ready  
✅ **Works with SSO** - maintains single sign-on across tabs  
✅ **Prevents race conditions** between React and MSAL initialization  

## Testing

To verify the fix works:

1. **Login to the application**
2. **Refresh the page (F5 or Cmd+R)**
3. **Expected result:** Brief loading screen → App loads without requiring re-login
4. **Open in new tab:** Should also maintain login state (SSO)

## Additional Notes

- The 500ms delay is conservative and ensures MSAL has time to read from localStorage
- The `inProgress` check catches MSAL's own initialization states ('startup', 'handleRedirect')
- This fix works in conjunction with the existing `localStorage` cache configuration in `authConfig.js`
- The solution is compatible with both popup and redirect authentication flows
