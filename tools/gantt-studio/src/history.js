// A gesture is recorded on save, not on pointer frames. Histories never cross projects.
export class ProjectHistory {
  constructor(limit = 100) {
    this.limit = limit;
    this.projects = new Map();
  }
  entry(id, state) {
    if (!this.projects.has(id))
      this.projects.set(id, {
        past: [],
        future: [],
        present: structuredClone(state),
      });
    return this.projects.get(id);
  }
  record(id, state) {
    const h = this.entry(id, state);
    if (JSON.stringify(h.present) === JSON.stringify(state)) return;
    h.past.push(h.present);
    if (h.past.length > this.limit) h.past.shift();
    h.present = structuredClone(state);
    h.future = [];
  }
  undo(id) {
    const h = this.projects.get(id);
    if (!h?.past.length) return null;
    h.future.push(h.present);
    h.present = h.past.pop();
    return structuredClone(h.present);
  }
  redo(id) {
    const h = this.projects.get(id);
    if (!h?.future.length) return null;
    h.past.push(h.present);
    h.present = h.future.pop();
    return structuredClone(h.present);
  }
  status(id) {
    const h = this.projects.get(id);
    return { undo: !!h?.past.length, redo: !!h?.future.length };
  }
}
