# IntelliCode Troubleshooting Guide

## Common Issues and Solutions

### 1. IntelliCode Not Activating
If you see "Sorry, something went wrong activating IntelliCode support for Python":

**Solution:**
- Reload VS Code Window (Ctrl+Shift+P → "Reload Window")
- Check Output → Python and VS IntelliCode for specific errors
- Ensure Python extension is properly installed and activated

### 2. Virtual Environment Not Detected
IntelliCode needs to know which Python environment to use.

**Current Configuration:**
```json
{
    "python.pythonPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
    "python.venvPath": "${workspaceFolder}\\.venv"
}
```

**To switch to ml_venv (in Macine learing (bird deaseser)):**
```json
{
    "python.pythonPath": "${workspaceFolder}\\Macine learing (bird deaseser)\\ml_venv\\Scripts\\python.exe",
    "python.venvPath": "${workspaceFolder}\\Macine learing (bird deaseser)\\ml_venv"
}
```

### 3. Manual IntelliCode Indexing
Sometimes IntelliCode needs manual triggering:

1. Open Command Palette (Ctrl+Shift+P)
2. Type "IntelliCode: Update Database"
3. Wait for completion (may take a few minutes)

### 4. Checking Logs
To diagnose issues:
- View → Output → Select "Python" from dropdown
- View → Output → Select "VS IntelliCode" from dropdown
- Look for error messages or warnings

### 5. Extension Conflicts
Ensure these extensions are installed and enabled:
- Microsoft Python Extension
- VS IntelliCode
- Pylance (for best IntelliSense experience)

### 6. Workspace Trust
If you see workspace trust issues:
- Click on the workspace trust indicator in the bottom-left
- Select "Trust Workspace"

### 7. Specific to This Workspace
This workspace contains:
- Main project: `c:\Users\Ali\OneDrive\Belgeler\pyton` (using `.venv`)
- Sub-project: `Macine learing (bird deaseser)` (has `ml_venv`)

If working primarily in the sub-project, consider:
1. Opening that folder as the main workspace
2. Or updating the python.pythonPath to point to ml_venv

## Quick Fixes to Try Now

1. **Reload Window**: Ctrl+Shift+P → "Reload Window"
2. **Check Python Path**: 
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose the correct `.venv\Scripts\python.exe`
3. **Trigger IntelliCode**:
   - Ctrl+Shift+P → "IntelliCode: Update Database"
4. **Check Output Windows** for specific error messages
5. **Restart VS Code** completely

## Verification
After applying fixes, you should see:
- IntelliSense working (auto-completion)
- Parameter hints
- Code suggestions
- No error messages in the IntelliCode output window