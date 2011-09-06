# Defines a platform specific view script
# Mine depends on a modified version of the Mac Preview app
 
"""
Make SURE you backup your Preview.app first, and don't try this
if you are uncomfortable with sudo, or unwilling to restore your system
from backup
sudo defaults write /Applications/Preview.app/Contents/Info NSAppleScriptEnabled -bool true
sudo chmod 644 /Applications/Preview.app/Contents/Info.plist
sudo codesign -f -s - /Applications/Preview.app
"""

import os
from appscript import *
from mactypes import *
ph = None
def view(fname):
    global ph
    if ph is None:
        ph = app('Preview')
    ph.open(Alias(os.path.abspath(fname)))

def test_view():
    view("test.jpg")
    
if __name__ == "__main__":
    test_view()
