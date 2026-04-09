use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde_json::Value;
use tauri::{Emitter, Manager};

struct SidecarState {
    stdin: Mutex<Option<std::process::ChildStdin>>,
    child: Mutex<Option<Child>>,
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
    // Find project root: try exe dir ancestors, then env var, then cwd heuristics
    let project_root = std::env::var("VOICE_ASSISTANT_ROOT")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| {
            // If exe is in ui/src-tauri/target/release/, walk up to project root
            std::env::current_exe().ok().and_then(|exe| {
                let mut dir = exe.parent()?.to_path_buf();
                // Walk up until we find pyproject.toml
                for _ in 0..6 {
                    if dir.join("pyproject.toml").exists() {
                        return Some(dir);
                    }
                    dir = dir.parent()?.to_path_buf();
                }
                None
            })
        })
        .or_else(|| {
            // Try cwd and cwd/..
            let cwd = std::env::current_dir().ok()?;
            if cwd.join("pyproject.toml").exists() {
                return Some(cwd);
            }
            let parent = cwd.parent()?;
            if parent.join("pyproject.toml").exists() {
                return Some(parent.to_path_buf());
            }
            None
        })
        .unwrap_or_else(|| std::path::PathBuf::from("/home/winers/voice-assistant"));

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

    // Read stdout/stderr before moving child into state
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    app.manage(SidecarState {
        stdin: Mutex::new(Some(stdin)),
        child: Mutex::new(Some(child)),
    });

    // Read stdout → parse JSON → emit to webview
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
                    child: Mutex::new(None),
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
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // Kill sidecar and its children on app exit
                let state: tauri::State<SidecarState> = app.state();

                // Close stdin first — signals Python to exit gracefully
                if let Ok(mut stdin_lock) = state.stdin.lock() {
                    stdin_lock.take();
                }

                // Kill the child process if it hasn't exited
                if let Ok(mut child_lock) = state.child.lock() {
                    if let Some(mut child) = child_lock.take() {
                        eprintln!("[tauri] Killing sidecar PID {}", child.id());
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }

                eprintln!("[tauri] Cleanup complete");
            }
        });
}
