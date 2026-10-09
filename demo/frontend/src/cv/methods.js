export async function fetchMethods() {
  const r = await fetch('/api/methods');
  if (!r.ok) throw new Error('methods: ' + r.status);
  return r.json();
}

export function visibleMethods(meta) {
  if (!meta) return [];
  return Object.entries(meta)
    .filter(([_, m]) => !m.hidden && m.available)
    .map(([id, m]) => ({ id, label: id, ...m }));
}