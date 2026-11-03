"""Desktop check: a result computed in one GUI session is reused after reopening the GUI.

    uv run --locked python tests/reopen_check.py          # runs both phases in separate processes

Phase 1 (new process): open the GUI, compute S01 for the first recording, close.
Phase 2 (new process): reopen with the same preferences; S01 must show as stored and
be displayed from storage without starting a computation.
Uses its own preferences file and output folder; the user's settings are untouched.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.02)


def phase(name, prefs, output):
    from data_analysis_gui import config as C
    C.PREFERENCES_PATH = Path(prefs)
    from data_analysis_gui.gui.app import App
    app = App()
    if name == '1':
        app.output_root.set(output)
    pump(app, 2.0)
    app.set_mode('single')
    view = app.single_view
    view.rec_tree.selection_set(view.rec_tree.get_children()[0])
    view.sec_tree.selection_set('s01')
    deadline = time.time() + 60
    while view.status_of.get('s01') is None and time.time() < deadline:
        pump(app, 0.2)
    status = view.status_of.get('s01', ('none',))[0]
    started = False
    if name == '1':
        view.compute_current()
        started = app.process is not None
        while app.process and app.process.poll() is None:
            pump(app, 0.3)
        pump(app, 4.0)
        status = view.status_of.get('s01', ('none',))[0]
    tabs = [view.result.notebook.tab(t, 'text') for t in view.result.notebook.tabs()]
    result = dict(phase=name, status=status, worker_started=started or app.process is not None, tabs=len(tabs),
                  output=app.output_root.get())
    app.close()
    print('@@', json.dumps(result))


def main():
    if len(sys.argv) > 1:
        phase(*sys.argv[1:4])
        return
    work = Path(__file__).resolve().parents[1] / '.local' / f'reopen_{int(time.time())}'
    work.mkdir(parents=True)
    prefs, output = work / 'preferences.json', work / 'output'
    results = []
    for name in ('1', '2'):
        out = subprocess.run([sys.executable, __file__, name, str(prefs), str(output)], capture_output=True,
                             text=True, encoding='utf-8', errors='replace')
        line = next((l for l in out.stdout.splitlines() if l.startswith('@@')), None)
        if line is None:
            print(out.stdout[-2000:], out.stderr[-2000:])
            raise SystemExit(f'phase {name} failed')
        results.append(json.loads(line[3:]))
        print(results[-1])
    assert results[0]['status'] == 'cached', results[0]
    assert results[1]['status'] == 'cached' and not results[1]['worker_started'] and results[1]['tabs'] >= 3, results[1]
    assert results[1]['output'] == str(output)
    print('OK: reopened GUI reuses the stored result without recomputing')


if __name__ == '__main__':
    main()
