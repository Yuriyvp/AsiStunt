/**
 * Hook for subscribing to specific signal types from the Python backend.
 */
import { useEffect, useState, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';

export function useSignals(signalTypes = []) {
  const [signals, setSignals] = useState({});
  const typesRef = useRef(signalTypes);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event === 'signal' && typesRef.current.includes(data.type)) {
        setSignals(prev => ({ ...prev, [data.type]: data }));
      }
    });

    return () => { unlisten.then(fn => fn()); };
  }, []);

  return signals;
}
