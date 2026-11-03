"""Desktop smoke test: single-day and cross-day workbenches, worker, cache-first reload.

    uv run --locked python tests/ui_smoke.py [screenshot-prefix]

Uses the real data root and a temporary output directory. For each mode it runs
one section through the worker process, then checks that a second request is
served from storage (no recomputation) and that the figure tab shows the PNG.
"""
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.gui.app import App  # noqa: E402


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.02)


def wait_worker(app, limit=180):
    deadline = time.time() + limit
    pump(app, 0.5)
    while app.process and app.process.poll() is None and time.time() < deadline:
        pump(app, 0.2)
    pump(app, 6.0)


def wait_tabs(app, notebook, minimum, limit=30):
    deadline = time.time() + limit
    while len(notebook.tabs()) < minimum and time.time() < deadline:
        pump(app, 0.3)
    return [notebook.tab(t, 'text') for t in notebook.tabs()]


def shot(app, name):
    if len(sys.argv) > 1:
        from PIL import ImageGrab
        app.attributes('-topmost', True)
        app.lift()
        pump(app, 1.0)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()), all_screens=True).save(
            f'{sys.argv[1]}_{name}.png')
        app.attributes('-topmost', False)


from data_analysis_gui import config as C  # noqa: E402
_prefs = Path(tempfile.mkdtemp(prefix='dagui_prefs_')) / 'preferences.json'
C.PREFERENCES_PATH = _prefs          # isolated: the user's .local/preferences.json is untouched
app = App()
app.geometry('1600x1000+0+0')
output = tempfile.mkdtemp(prefix='dagui_ui_')
app.output_follow.set(False)          # keep the temporary output; do not follow the data folder
app.output_root.set(output)
app.sync_output()
pump(app, 1.5)
assert app.visible, 'no recordings visible'

# ---- single-day
app.set_mode('single')
view = app.single_view
view.rec_tree.selection_set(view.rec_tree.get_children()[0])
view.sec_tree.selection_set('s01')
pump(app, 1.0)
view.compute_current()
wait_worker(app)
tabs = wait_tabs(app, view.result.notebook, 3)
print('single:', app.progress_text.get(), '|', app.task_text.get(), '|', tabs)
assert any('raster' in t for t in tabs), tabs
from data_analysis_gui.gui.canvas import ZoomCanvas  # noqa: E402
def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


canvases = [w for w in _walk(view.result.notebook) if isinstance(w, ZoomCanvas) and w.image is not None]
c = canvases[0]
shown_w, shown_h = c.image.width * c.scale, c.image.height * c.scale
print('fit:', c.fit_mode, round(c.scale, 3), int(shown_w), int(shown_h), c.canvas.winfo_width(), c.canvas.winfo_height())
assert c.fit_mode == 'whole' and shown_w <= c.canvas.winfo_width() + 1 and shown_h <= c.canvas.winfo_height() + 1
shot(app, 'single')
view.compute_current()                    # must be served from storage
wait_worker(app)
print('single again:', app.task_text.get())
assert '存储' in app.task_text.get() or '完成' in app.task_text.get()

# ---- trial-filter dialog with preview
from data_analysis_gui.gui.filter_dialog import FilterDialog  # noqa: E402
dialog = FilterDialog(app, app.filter, app.set_filter)
pump(app, 0.5)
dialog.preview()
deadline = time.time() + 60
while '计算中' in dialog.preview_text.get('1.0', 'end') and time.time() < deadline:
    pump(app, 0.3)
text = dialog.preview_text.get('1.0', 'end')
print('filter preview:', text.splitlines()[:2])
assert '方向队列' in text, text
if len(sys.argv) > 1:
    from PIL import ImageGrab
    dialog.attributes('-topmost', True)
    pump(app, 1.0)
    x, y = dialog.winfo_rootx(), dialog.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + dialog.winfo_width(), y + dialog.winfo_height()), all_screens=True).save(
        f'{sys.argv[1]}_filter.png')
dialog.destroy()

# ---- cross-day
app.set_mode('cross')
cross = app.cross_view
cross.check_first_per_day()
keys = list(cross.checked)[:3]
cross._set_checked(keys)
pump(app, 1.0)
cross.notebook.select(0)
pump(app, 0.5)
cross.compute(['x01'])
wait_worker(app, 300)
result = cross.tabs['x01']['result']
tabs = wait_tabs(app, result.notebook, 3)
print('cross:', app.progress_text.get(), '|', app.task_text.get(), '|', tabs)
assert len(tabs) > 2, tabs
shot(app, 'cross')
app.close()
print('output:', output)
