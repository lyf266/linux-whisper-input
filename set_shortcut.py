import dbus
import sys

try:
    bus = dbus.SessionBus()
    kglobalaccel = bus.get_object('org.kde.kglobalaccel', '/kglobalaccel')
    iface = dbus.Interface(kglobalaccel, 'org.kde.KGlobalAccel')
    flags = dbus.UInt32(0)
    
    # Define the exact 4-element actionIds
    legacy_action_id = ['voice-input', '_launch', 'voice-input', '启动语音输入']
    desktop_action_id = ['voice-input.desktop', '_launch', '语音输入', '语音输入']
    
    # 1. Clear both to prevent conflict
    print("Clearing legacy shortcut and new shortcut...")
    iface.setShortcut(legacy_action_id, [], flags)
    iface.setShortcut(desktop_action_id, [], flags)
    print("Shortcuts cleared.")
    
    # 2. Assign Meta+H (keycode 268435528) to voice-input.desktop
    target_keys = [268435528]
    print("\nAssigning Meta+H to voice-input.desktop...")
    result = iface.setShortcut(desktop_action_id, target_keys, flags)
    print(f"Result (assigned keycodes): {list(result)}")
    
    # Verify by reading it back
    current_shortcut = iface.shortcut(desktop_action_id)
    print(f"Current registered shortcut for desktop: {list(current_shortcut)}")
    
    if 268435528 in current_shortcut:
        print("\n🎉 Success! Meta+H is now successfully bound to voice-input.desktop in KDE!")
        sys.exit(0)
    else:
        print("\n❌ Error: Failed to bind Meta+H. The shortcut was rejected by the system.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error during D-Bus execution: {e}")
    sys.exit(1)
