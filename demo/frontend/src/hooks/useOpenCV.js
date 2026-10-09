import { useEffect, useState } from 'react';

export function useOpenCV() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (window.cv?.Mat) { setReady(true); return; }
    const id = setInterval(() => {
      if (window.cv?.Mat) { setReady(true); clearInterval(id); }
    }, 100);
    return () => clearInterval(id);
  }, []);
  return ready;
}