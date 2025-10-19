# Finding Cognitive Services Permissions in Azure Portal - Visual Guide

## Problem: Can't Find "Cognitive Services OpenAI Contributor" Role

This guide shows you **exactly where** to find the Cognitive Services permissions in Azure Portal.

---

## Step-by-Step Instructions with Screenshots Description

### Step 1: Navigate to Your Azure OpenAI Resource

```
Azure Portal (portal.azure.com)
│
├─ Search bar at top: Type "Azure OpenAI" or your resource name
└─ Click on your Azure OpenAI resource
```

**What you should see:**
- Resource overview page with Details, Essentials, Properties

---

### Step 2: Open Access Control (IAM)

```
Your Azure OpenAI Resource Page
│
Left Sidebar:
├─ Overview
├─ Activity log
├─ Access control (IAM)  ← CLICK HERE
├─ Tags
├─ Diagnose and solve problems
└─ Settings
    ├─ Keys and Endpoint
    └─ ...
```

**What you should see:**
- A page titled "Access control (IAM)"
- Tabs: "Check access", "Role assignments", "Deny assignments", "Classic administrators"
- A button bar at the top with "+ Add" button

---

### Step 3: Click "+ Add" Button

```
Access control (IAM) page
│
Top button bar:
├─ [+ Add] ← CLICK HERE
├─ Remove
└─ Refresh
│
Dropdown appears:
├─ Add role assignment    ← SELECT THIS
├─ Add deny assignment
└─ Add co-administrator (classic)
```

---

### Step 4: Find Cognitive Services Roles

```
Add role assignment wizard
│
Tab 1: [Role] ← You are here
│
├─ Search box: "Search by role name or description"
│  └─ Type: "Cognitive Services" or "OpenAI"
│
Results shown below:
├─ [●] Cognitive Services Contributor
├─ [●] Cognitive Services OpenAI Contributor  ← SELECT THIS ONE
├─ [●] Cognitive Services OpenAI User
├─ [●] Cognitive Services User
└─ ... (more roles)

Description panel on right shows:
"Allows full access to Azure OpenAI endpoints..."

[Next] button at bottom
```

**IMPORTANT:** The roles are in **alphabetical order**. Scroll down if needed!

---

### Step 5: Add Members

```
Tab 2: [Members] ← You are now here

Select members
├─ [+ Select members] button ← CLICK THIS
│
Side panel opens:
│
Select members
├─ Search box
│  └─ Type: user email or name
│
Search results:
├─ [ ] John Doe (john@company.com)
├─ [ ] Jane Smith (jane@company.com)  ← CHECK USERS
└─ [ ] App Service (managed identity)

Selected members: (shown at bottom)
└─ Jane Smith

[Select] button at bottom
```

---

### Step 6: Review and Assign

```
Tab 3: [Review + assign]

Review details:
├─ Role: Cognitive Services OpenAI Contributor
├─ Scope: Resource
├─ Resource: your-openai-resource
└─ Members: jane@company.com

[Review + assign] button at bottom ← CLICK THIS
```

**Done!** User now has access.

---

## Troubleshooting Guide

### Issue 1: I don't see "Access control (IAM)" in the left menu

**Cause:** You might not have sufficient permissions.

**Solution:**
1. Check if you have Owner or User Access Administrator role on the subscription
2. Ask your Azure administrator to grant you permissions
3. Alternative: Use Azure CLI (see below)

---

### Issue 2: The role list is empty or doesn't show Cognitive Services roles

**Cause:** Filter might be applied or wrong scope.

**Solutions:**
1. **Clear any filters**: Look for filter buttons/dropdowns and reset them
2. **Try different tabs**: Click on "Job function roles" tab instead of "All roles"
3. **Scroll down**: There are 100+ roles, keep scrolling
4. **Type in search**: Type "Cognitive" or "OpenAI" in the search box
5. **Check scope**: Make sure you're on the OpenAI resource, not subscription level

---

### Issue 3: I can add role but user still can't access

**Causes and Solutions:**

1. **Permission propagation delay**
   - Wait 5-10 minutes for Azure to propagate permissions
   - Ask user to log out and log back in

2. **Wrong scope assigned**
   - Verify: Azure Portal → Resource → Access control (IAM) → Role assignments
   - The scope should be the OpenAI resource, not subscription

3. **Application using API Key instead of RBAC**
   - If your app uses `AZURE_OPENAI_API_KEY`, RBAC doesn't apply
   - RBAC only works with Azure AD authentication

---

## Alternative Method: Using Azure CLI

If the portal is not working, use Azure CLI:

### 1. List Available Roles

```bash
az role definition list \
  --query "[?contains(roleName, 'Cognitive')].{Name:roleName, Description:description}" \
  --output table
```

**You should see:**
```
Name                                      Description
----------------------------------------  -----------------------------------
Cognitive Services Contributor            Full access to Cognitive Services
Cognitive Services OpenAI Contributor     Allows full access to Azure OpenAI...
Cognitive Services OpenAI User            Read access to Azure OpenAI...
Cognitive Services User                   Allows read access...
```

### 2. Get Resource ID

```bash
# Get your OpenAI resource ID
az cognitiveservices account show \
  --name YOUR_OPENAI_RESOURCE_NAME \
  --resource-group YOUR_RESOURCE_GROUP \
  --query id \
  --output tsv
```

**Output example:**
```
/subscriptions/12345.../resourceGroups/myRG/providers/Microsoft.CognitiveServices/accounts/myOpenAI
```

### 3. Assign Role

```bash
# Assign role to user
az role assignment create \
  --assignee user@company.com \
  --role "Cognitive Services OpenAI Contributor" \
  --scope "/subscriptions/YOUR_SUB/resourceGroups/YOUR_RG/providers/Microsoft.CognitiveServices/accounts/YOUR_OPENAI"
```

**Success output:**
```json
{
  "principalName": "user@company.com",
  "roleDefinitionName": "Cognitive Services OpenAI Contributor",
  "scope": "/subscriptions/.../accounts/myOpenAI",
  "type": "Microsoft.Authorization/roleAssignments"
}
```

### 4. Verify Assignment

```bash
# List role assignments for the resource
az role assignment list \
  --scope "/subscriptions/YOUR_SUB/resourceGroups/YOUR_RG/providers/Microsoft.CognitiveServices/accounts/YOUR_OPENAI" \
  --output table
```

---

## Quick Reference: All Cognitive Services Roles

| Role Name | What It Does | Use Case |
|-----------|--------------|----------|
| **Cognitive Services OpenAI Contributor** | ✅ Can use API, manage deployments | **RECOMMENDED** for users who need to chat/use AI |
| Cognitive Services OpenAI User | Read-only access | For monitoring/viewing only |
| Cognitive Services Contributor | Full access to all Cognitive Services | For admins managing multiple services |
| Cognitive Services User | Read access to all Cognitive Services | For basic read-only access |

---

## Still Can't Find It?

### Check Your Azure Portal Version

Azure Portal sometimes shows different interfaces:

1. **New Experience (2024+)**:
   - IAM → Add → Add role assignment → Search for "Cognitive Services"

2. **Classic Experience**:
   - IAM → Add role assignment → Role dropdown → Search for "Cognitive Services"

### Check Your Browser

- Try a different browser (Edge, Chrome, Firefox)
- Clear browser cache
- Use Incognito/Private mode

### Check Your Azure Environment

```bash
# Verify you're in the right subscription
az account show --query "{Name:name, Id:id}" --output table

# Verify the OpenAI resource exists
az cognitiveservices account list --query "[?kind=='OpenAI'].{Name:name, Location:location}" --output table
```

---

## Contact Support

If none of these work:

1. **Azure Portal**: Click the "?" icon (help) → "Contact support"
2. **Azure Community**: https://techcommunity.microsoft.com/azure
3. **Azure Support**: https://azure.microsoft.com/support/

---

## Summary

**The most common location:**
```
Azure Portal
→ Your Azure OpenAI Resource  
→ Access control (IAM) [left menu]
→ + Add [top button]
→ Add role assignment
→ Search: "Cognitive Services OpenAI Contributor"
→ Select role → Next
→ + Select members → Add users → Select
→ Review + assign
```

**Role to select:** `Cognitive Services OpenAI Contributor`

**Where users see it:** In the **Role** tab of the Add role assignment wizard, using the **search box**.

Good luck! 🎉
