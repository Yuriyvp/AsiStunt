use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use serde_json::Value;
use tauri::{Emitter, Manager};

struct SidecarState {
    stdin: Mutex<Option<std::process::ChildStdin>>,
}

#[tauri::command]
fn send_command(state: tauri::State<SidecarState>, cmd: String) -> Result<(), String> {
    let mut stdin_lock = state.stdin.lock().map_err(|e| e.to_string())?;
    if let Some(stdin) = stdin_lock.as_mut() {
        writeln!(stdin, "{}", cmd).map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;
        Ok(())
    } else {
        Err("Sidecar not running".into())
    }
}

fn spawn_python_sidecar(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // Find project root (ui/ parent)
    let project_root = std::env::current_dir()
        .unwrap_or_default()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(".."));

    let venv_python = project_root.join(".venv/bin/python");
    let python_bin = if venv_python.exists() {
        venv_python.to_string_lossy().to_string()
    } else {
        "python3".to_string()
    };

    eprintln!("[tauri] Spawning sidecar: {} -u -m voice_assistant.main", python_bin);
    eprintln!("[tauri] Working directory: {}", project_root.display());

    let mut child = Command::new(&python_bin)
        .args(["-u", "-m", "voice_assistant.main"])
        .current_dir(&project_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn Python sidecar: {}", e))?;

    eprintln!("[tauri] Sidecar PID: {}", child.id());

    let stdin = child.stdin.take().unwrap();
    app.manage(SidecarState {
        stdin: Mutex::new(Some(stdin)),
    });

    // Read stdout → parse JSON → emit to webview
    let stdout = child.stdout.take().unwrap();
    let handle = app.handle().clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(&line) {
                        Ok(msg) => {
                            let _ = handle.emit("python_event", &msg);
                        }
                        Err(_) => {
                            eprintln!("[python:stdout] {}", line);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[tauri] stdout read error: {}", e);
                    break;
                }
            }
        }
        eprintln!("[tauri] Sidecar stdout closed");
    });

    // Read stderr → forward to Tauri debug console
    let stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[python] {}", line);
            }
        }
        eprintln!("[tauri] Sidecar stderr closed");
    });

    Ok(())
}

fn setup_global_shortcuts(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(desktop)]
    {
        use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

        let mute_toggle = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
        let debug_toggle = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyD);
        let compact_toggle = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyC);

        let handle = app.handle().clone();
        app.handle().plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |_app, shortcut, event| {
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }
                    if shortcut == &mute_toggle {
                        let _ = handle.emit("shortcut", "mute_toggle");
                    } else if shortcut == &debug_toggle {
                        let _ = handle.emit("shortcut", "debug_toggle");
                    } else if shortcut == &compact_toggle {
                        let _ = handle.emit("shortcut", "compact_toggle");
                    }
                })
                .build(),
        )?;

        let gs = app.global_shortcut();
        let _ = gs.register(mute_toggle);
        let _ = gs.register(debug_toggle);
        let _ = gs.register(compact_toggle);

        eprintln!("[tauri] Global shortcuts registered: Ctrl+Shift+Space/D/C");
    }
    Ok(())
}

fn setup_system_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::TrayIconBuilder;

    let new_conv = MenuItem::with_id(app, "new_conversation", "New Conversation", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
    let debug = MenuItem::with_id(app, "debug", "Debug Window", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&new_conv, &settings, &debug, &quit])?;

    let handle = app.handle().clone();
    TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Voice Assistant")
        .on_menu_event(move |_app, event| {
            let id = event.id().as_ref();
            match id {
                "quit" => {
                    let _ = handle.emit("tray_action", "quit");
                    std::process::exit(0);
                }
                "new_conversation" => {
                    let _ = handle.emit("tray_action", "new_conversation");
                }
                "settings" => {
                    let _ = handle.emit("tray_action", "settings");
                }
                "debug" => {
                    let _ = handle.emit("tray_action", "debug");
                }
                _ => {}
            }
        })
        .build(app)?;

    eprintln!("[tauri] System tray created");
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Spawn Python sidecar
            if let Err(e) = spawn_python_sidecar(app) {
                eprintln!("[tauri] WARNING: Sidecar spawn failed: {}", e);
                // Still run UI — sidecar can be started later
                app.manage(SidecarState {
                    stdin: Mutex::new(None),
                });
            }

            // Global shortcuts
            if let Err(e) = setup_global_shortcuts(app) {
                eprintln!("[tauri] WARNING: Global shortcuts failed: {}", e);
            }

            // System tray
            if let Err(e) = setup_system_tray(app) {
                eprintln!("[tauri] WARNING: System tray failed: {}", e);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![send_command])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
