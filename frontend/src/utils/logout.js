// NOTE: All custom logout coordination functions have been REMOVED.
//
// According to Microsoft's official documentation:
// https://learn.microsoft.com/en-us/entra/msal/javascript/browser/logout
// 
// "The logout process for MSAL takes two steps:
//  1. Clear the MSAL cache.
//  2. Clear the session on the identity server."
// 
// Both steps are handled automatically by logoutRedirect() and logoutPopup().
// 
// Use MSAL's built-in logout methods:
//   instance.logoutRedirect() or instance.logoutPopup()
//
// If you need to clear cache without server logout, use:
//   instance.logoutRedirect({ onRedirectNavigate: () => false })

