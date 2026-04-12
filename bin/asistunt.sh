#!/bin/bash
export VOICE_ASSISTANT_ROOT="/home/winers/voice-assistant"
cd "$VOICE_ASSISTANT_ROOT"
source .venv/bin/activate
export PATH="$VOICE_ASSISTANT_ROOT/bin:$PATH"

# Clean VS Code snap environment pollution.
# Running inside a VS Code snap terminal injects snap-specific GTK/GDK/GIO
# paths that cause WebKitNetworkProcess to load incompatible snap core20 glibc,
# crashing with "undefined symbol: __libc_pthread_init".
unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE
unset GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR
unset GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR
unset SNAP SNAP_NAME SNAP_REVISION SNAP_ARCH SNAP_VERSION
unset SNAP_COMMON SNAP_DATA SNAP_USER_DATA SNAP_USER_COMMON
unset SNAP_REAL_HOME SNAP_INSTANCE_NAME SNAP_CONTEXT SNAP_COOKIE
unset SNAP_EUID SNAP_UID SNAP_LAUNCHER_ARCH_TRIPLET
export XDG_DATA_DIRS="/usr/share/ubuntu:/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop"
export XDG_CONFIG_DIRS="/etc/xdg/xdg-ubuntu:/etc/xdg"
export XDG_DATA_HOME="$HOME/.local/share"

exec "$VOICE_ASSISTANT_ROOT/ui/src-tauri/target/release/ui"
