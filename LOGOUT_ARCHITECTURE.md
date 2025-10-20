# Logout Architecture Diagram

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Application                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │  ChatHeader.jsx  │              │   Sidebar.jsx    │            │
│  ├──────────────────┤              ├──────────────────┤            │
│  │ - Logout button  │              │ - Sign out btn   │            │
│  │ - User info      │              │ - Config cards   │            │
│  └────────┬─────────┘              └────────┬─────────┘            │
│           │                                  │                       │
│           │ uses                    uses     │                       │
│           ▼                                  ▼                       │
│  ┌────────────────────────────────────────────────────┐            │
│  │         useLogout Hook (useLogout.js)              │            │
│  ├────────────────────────────────────────────────────┤            │
│  │ - logout()        : Async logout function          │            │
│  │ - isLoggingOut    : Loading state boolean          │            │
│  │ - logoutType      : 'popup' | 'redirect'           │            │
│  │ - postLogoutRedirectUri : string                   │            │
│  └────────┬───────────────────────────────────────────┘            │
│           │ uses                                                     │
│           ▼                                                          │
│  ┌────────────────────────────────────────────────────┐            │
│  │       Logout Utilities (logout.js)                 │            │
│  ├────────────────────────────────────────────────────┤            │
│  │ - setGlobalLogoutFlags()                           │            │
│  │ - checkGlobalLogoutFlag(maxAgeMs)                  │            │
│  │ - markLogoutFlagProcessed(logoutTime)              │            │
│  │ - clearMSALStorage()                               │            │
│  └────────┬───────────────────────────────────────────┘            │
│           │                                                          │
└───────────┼──────────────────────────────────────────────────────────┘
            │
            │ reads/writes
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Browser Storage Layer                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    localStorage                               │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ - msal_global_logout            : timestamp                  │  │
│  │ - msal_global_logout_processed  : timestamp                  │  │
│  │ - app_logged_in                 : '1' | null                 │  │
│  │ - msal.* keys                   : MSAL tokens & cache        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    document.cookie                            │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ - msal_global_logout  : timestamp (max-age: 300s)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────┬───────────────────────────────────────────────────────────┘
            │
            │ monitored by
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SSO Showcase App (Port 8001)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Vanilla JavaScript (index.html)                  │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  Logout Button Handler                              │    │  │
│  │  ├─────────────────────────────────────────────────────┤    │  │
│  │  │  1. Set localStorage.msal_global_logout             │    │  │
│  │  │  2. Set localStorage.msal_global_logout_processed   │    │  │
│  │  │  3. Set document.cookie.msal_global_logout          │    │  │
│  │  │  4. Clear localStorage.app_logged_in                │    │  │
│  │  │  5. Call msalInstance.logoutRedirect()              │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  Periodic Check (setInterval 5s)                    │    │  │
│  │  ├─────────────────────────────────────────────────────┤    │  │
│  │  │  1. Check localStorage.msal_global_logout           │    │  │
│  │  │  2. Compare with msal_global_logout_processed       │    │  │
│  │  │  3. If recent & not processed:                      │    │  │
│  │  │     - Update UI to logged out state                 │    │  │
│  │  │     - Clear MSAL localStorage keys                  │    │  │
│  │  │     - Mark as processed                             │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Logout Flow Sequence

### Scenario: User clicks logout in React App (localhost:5173)

```
User                ChatHeader/Sidebar      useLogout Hook      logout.js           Browser Storage      SSO App (8001)
 |                         |                      |                  |                      |                    |
 |---- Click Logout ------>|                      |                  |                      |                    |
 |                         |                      |                  |                      |                    |
 |                         |---- logout() ------->|                  |                      |                    |
 |                         |                      |                  |                      |                    |
 |                         |                      |-- setGlobalLogoutFlags() -->            |                    |
 |                         |                      |                  |                      |                    |
 |                         |                      |                  |--- Set msal_global_logout -->             |
 |                         |                      |                  |--- Set msal_global_logout_processed -->   |
 |                         |                      |                  |--- Set cookie --->   |                    |
 |                         |                      |                  |--- Remove app_logged_in ->                |
 |                         |                      |                  |<------ OK ----------|                    |
 |                         |                      |                  |                      |                    |
 |                         |                      |--- logoutPopup/Redirect() ----------->  |                    |
 |                         |                      |                  |                      |                    |
 |                         |                      |                  |    [Periodic Check every 5s]              |
 |                         |                      |                  |                      |<--- Check flags ---|
 |                         |                      |                  |                      |                    |
 |                         |                      |                  |                      |--- Logout detected ->
 |                         |                      |                  |                      |                    |
 |                         |                      |                  |                      |<--- Update UI -----|
 |                         |                      |                  |                      |<--- Clear storage -|
 |                         |                      |                  |                      |                    |
 |<---- Redirect to / -----|<--------------------|                  |                      |                    |
 |                         |                      |                  |                      |                    |
```

### Scenario: User clicks logout in SSO Showcase (localhost:8001)

```
User                SSO App (8001)         Browser Storage      React App (5173)    useLogout/Components
 |                         |                      |                    |                    |
 |---- Click Logout ------>|                      |                    |                    |
 |                         |                      |                    |                    |
 |                         |--- Set msal_global_logout -------------->  |                    |
 |                         |--- Set msal_global_logout_processed --->   |                    |
 |                         |--- Set cookie ------>|                    |                    |
 |                         |--- Remove app_logged_in ->                |                    |
 |                         |                      |                    |                    |
 |                         |--- msalInstance.logoutRedirect() ------->  |                    |
 |                         |                      |                    |                    |
 |                         |                      |    [Periodic Check in App.jsx]          |
 |                         |                      |<--- Check flags ---|                    |
 |                         |                      |                    |                    |
 |                         |                      |--- Logout detected ------------------>  |
 |                         |                      |                    |                    |
 |                         |                      |<--- clearMSALStorage() --------------|  |
 |                         |                      |<--- Update UI ----------------------|  |
 |                         |                      |<--- Mark processed -----------------|  |
 |                         |                      |                    |                    |
 |<---- Redirect ----------|                      |                    |                    |
 |                         |                      |                    |                    |
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Logout Trigger                            │
│         (User clicks logout in any application)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              setGlobalLogoutFlags()                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Generate timestamp                                 │  │
│  │ 2. localStorage.setItem('msal_global_logout', ts)    │  │
│  │ 3. localStorage.setItem('msal_global_logout_...', ts)│  │
│  │ 4. document.cookie = 'msal_global_logout=...'        │  │
│  │ 5. localStorage.removeItem('app_logged_in')          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MSAL Logout Execution                           │
│         (logoutPopup or logoutRedirect)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌─────────────────────┐    ┌─────────────────────┐
│   Same App          │    │   Other Apps        │
│   (Immediate)       │    │   (Periodic Check)  │
├─────────────────────┤    ├─────────────────────┤
│ - Logout complete   │    │ - Check every 5s    │
│ - Redirect to /     │    │ - Detect flag       │
│ - Clear UI state    │    │ - Update UI         │
└─────────────────────┘    │ - Clear storage     │
                           │ - Mark processed    │
                           └─────────────────────┘
```

## Storage Keys Lifecycle

```
Action                     msal_global_logout    msal_global_logout_processed    app_logged_in
─────────────────────────  ──────────────────    ────────────────────────────    ─────────────
Initial State              not set               not set                         not set

User Logs In               not set               not set                         '1'

User Logs Out (App A)      timestamp_A           timestamp_A                     removed
  ↓ (immediately)          (set)                 (set)                           (removed)

Other App Detects          timestamp_A           timestamp_A                     removed
  ↓ (within 5s)            (exists)              (exists)                        (removed)

Other App Processes        removed               timestamp_A                     removed
  ↓                        (cleaned up)          (updated)                       (removed)

After 5 Minutes            auto-expired          timestamp_A                     removed
                           (if not cleaned)      (persists)                      (removed)
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Initiates Logout                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Check if already│
                │ logging out?    │
                └────┬───────┬────┘
                     │       │
                 Yes │       │ No
                     │       │
                     ▼       ▼
            ┌──────────┐   ┌────────────────────┐
            │ Return   │   │ Set isLoggingOut   │
            │ Early    │   │ = true             │
            └──────────┘   └────────┬───────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ Try: Set Global     │
                        │      Logout Flags   │
                        └────────┬───────┬────┘
                                 │       │
                          Success│       │Error
                                 │       │
                                 ▼       ▼
                        ┌─────────────────────┐
                        │ Try: MSAL Logout    │
                        └────────┬───────┬────┘
                                 │       │
                          Success│       │Error
                                 │       │
                                 ▼       ▼
                        ┌─────────────────────┐
                        │ Finally:            │
                        │ isLoggingOut = false│
                        └────────┬────────────┘
                                 │
                                 ▼
                        ┌─────────────────────┐
                        │ Catch: Log error    │
                        │ Re-throw for caller │
                        └─────────────────────┘
```

## Component Interaction Matrix

```
Component/Utility     Uses Hook?   Uses Utils?   Exports    Purpose
──────────────────    ──────────   ───────────   ───────    ───────────────────────
ChatHeader.jsx        ✓            ✗             UI         Logout button in header
Sidebar.jsx           ✓            ✗             UI         Sign out in sidebar
useLogout.js          ✗            ✓             Hook       React logout hook
logout.js             ✗            ✗             Utils      Core logout functions
App.jsx               ✗            ✓             Main       Monitors logout flags
index.html (8001)     ✗            ✗             App        SSO showcase app

Legend: ✓ = Yes, ✗ = No
```

## Browser Compatibility

```
Feature                  Chrome    Firefox    Safari    Edge      Notes
─────────────────────    ──────    ───────    ──────    ────      ─────────────────
localStorage             ✓         ✓          ✓         ✓         Full support
document.cookie          ✓         ✓          ✓         ✓         Full support
setInterval              ✓         ✓          ✓         ✓         Full support
async/await              ✓         ✓          ✓         ✓         ES2017+
MSAL.js                  ✓         ✓          ✓         ✓         MSAL v2.38+

Cross-origin cookies     Limited   Limited    Limited   Limited   SameSite policy
```
