/**
 * Hook for communicating with the Python sidecar via Tauri IPC.
 * Sends commands as JSON over stdin, receives events from stdout.
 */
import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export function useSidecar() {
  const [state, setState] = useState('DISABLED');
  const [events, setEvents] = useState([]);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;

      if (data.event === 'state_change') {
        setState(data.state);
      }

      setEvents(prev => [...prev.slice(-100), data]);
    });

    return () => { unlisten.then(fn => fn()); };
  }, []);

  const sendCommand = useCallback(async (cmd) => {
    try {
      await invoke('send_command', { cmd: JSON.stringify(cmd) });
    } catch (e) {
      console.error('Failed to send command:', e);
    }
  }, []);

  return { state, events, sendCommand };
}
