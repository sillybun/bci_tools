"""Hidden-Tk checks for the shared single/cross-day result table pager."""
import tkinter as tk
from tkinter import ttk

import pytest

from data_analysis_gui import storage
from data_analysis_gui.gui.widgets import ResultView


@pytest.fixture(scope='module')
def root():
    # The application uses one Tk interpreter per process; reuse it in GUI tests too.
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def test_file_button_rejects_release_without_own_press(root):
    from data_analysis_gui.gui.widgets import FileButton
    opened = []
    window = tk.Toplevel(root)
    ordinary = ttk.Button(window, command=lambda: opened.append('ordinary'))
    guarded = FileButton(window, command=lambda: opened.append('guarded'))
    ordinary.pack()
    guarded.pack()
    window.update()
    try:
        # TButton considers entering while Button1 is down to be a press.
        # Reproduce this without a preceding ButtonPress on either button.
        for button in (ordinary, guarded):
            button.event_generate('<Enter>')
            button.event_generate('<Enter>', state=0x100)
            assert button.instate(['pressed'])
            button.event_generate('<ButtonRelease-1>')
        assert opened == ['ordinary']
        guarded.event_generate('<ButtonPress-1>')
        guarded.event_generate('<ButtonRelease-1>')
        assert opened == ['ordinary', 'guarded']
        guarded.event_generate('<<Invoke>>')
        assert opened == ['ordinary', 'guarded', 'guarded']
    finally:
        window.destroy()


def test_result_refresh_preserves_page_and_reloads_changed_metadata(root, tmp_path, monkeypatch):
    from copy import deepcopy
    from data_analysis_gui.gui import widgets
    from data_analysis_gui.gui.canvas import ZoomCanvas
    loaded, opened = [], []
    monkeypatch.setattr(ZoomCanvas, 'load', lambda self, path: loaded.append(path.name))
    monkeypatch.setattr(widgets, 'open_path', lambda path: opened.append(path.name))
    record = dict(signature='test', figures=[dict(title=f'Atlas {n}',
                  files={'png': f'{n}.png'}) for n in (1, 2)], tables={}, digests={'1.png': 'old'})
    view = ResultView(root)
    view.show(record, tmp_path)
    tab = view.nametowidget(view.notebook.tabs()[0])
    tab.series_step(1)
    view.show(deepcopy(record), tmp_path)
    assert tab.winfo_exists() and tab.series_state['index'] == 1
    assert loaded == ['1.png', '2.png'] and opened == []
    record['digests']['1.png'] = 'new'
    view.show(record, tmp_path)
    assert not tab.winfo_exists() and loaded[-1] == '1.png'
    view.message('Missing')
    tabs = view.notebook.tabs()
    view.message('Missing')
    assert view.notebook.tabs() == tabs
    view.destroy()


def test_atlas_pages_share_one_tab_and_open_combined_pdf(root, tmp_path, monkeypatch):
    from data_analysis_gui.gui import widgets
    from data_analysis_gui.gui.canvas import ZoomCanvas
    loaded, opened = [], []
    monkeypatch.setattr(ZoomCanvas, 'load', lambda self, path: loaded.append(path.name))
    monkeypatch.setattr(widgets, 'open_path', lambda path: opened.append(path.name))
    figures = [dict(title=f'ANOVA tuning curves - all channels {n}', files={
        'png':f'{n}.png', 'pdf':f'{n}.pdf', 'svg':f'{n}.svg', 'csv':f'{n}.csv',
        'series_pdf':'all_channels.pdf'}) for n in (1,2)]
    view = ResultView(root)
    view.show(dict(signature='test',figures=figures,tables={}),tmp_path)
    assert len(view.notebook.tabs()) == 2  # one figure series plus Summary
    tab = view.nametowidget(view.notebook.tabs()[0])
    buttons = {w.cget('text'):w for w in walk(tab) if isinstance(w,ttk.Button)}
    assert buttons['◀ Prev'].instate(['disabled'])
    buttons['Next ▶'].invoke()
    assert loaded[-1] == '2.png' and tab.series_state['index'] == 1
    assert buttons['Next ▶'].instate(['disabled'])
    buttons['整组 PDF'].invoke()
    assert opened == ['all_channels.pdf']
    buttons['◀ Prev'].invoke()
    assert loaded[-1] == '1.png'
    view.destroy()


def test_single_view_multi_selection_compute_and_preview(root, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from data_analysis_gui.gui.app import SLOW_SINGLE_SECTIONS, SingleView
    from data_analysis_gui.sections import sections
    jobs=[];lookups=[]
    app=SimpleNamespace(prefs={},status=tk.StringVar(master=root),
                        run_jobs=lambda batch,callback:jobs.extend(batch) or True)
    monkeypatch.setattr(SingleView,'check_status',lambda self,items,callback:lookups.extend(items))
    view=SingleView(root,app)
    records=[SimpleNamespace(key=f'r{i}',path=Path(f'run{i}.xdf'),date='2026-11-02',run=str(i),
                             data_type='CO8 online' if i==2 else 'CO8',task_mode='centerout',
                             online=i==2,has_fpga_clock=True,channel_config='configs.ini',label=f'Record {i}')
             for i in (1,2,3)]
    view.set_recordings(records)
    view.rec_tree.selection_set(['r1','r2']);view.rec_tree.focus('r2')
    view.sec_tree.selection_set('s01');view.on_section();view.on_recording()
    assert str(view.rec_tree.cget('selectmode'))=='extended'
    assert view.preview_recordings()==[records[1]]
    assert lookups and all(len(item[2])==1 for item in lookups)
    assert lookups[-1][2]==[records[1]]  # never sign a single-day section with two recordings
    view.compute_current()
    assert len(jobs)==2 and [entry[1] for entry in jobs]==[[records[0]],[records[1]]]
    jobs.clear();view.compute_all()
    assert all(s.id not in SLOW_SINGLE_SECTIONS for s, _, _ in jobs)
    assert view.rec_tree.item('r1', 'tags') == ('CO8',)
    assert view.rec_tree.item('r2', 'tags') == ('CO8 online',)
    view.include_slow_sections.set(True)
    jobs.clear();view.compute_all()
    assert {r.key for _,recs,_ in jobs for r in recs}=={'r1','r2'}
    assert sum(s.id=='s11' for s,_,_ in jobs)==1
    assert all(len(recs)==1 for _,recs,_ in jobs)
    assert len(jobs) == sum(1 for s in sections('single') for r in records[:2]
                            if 'online' not in s.requires or r.online)
    saved={};view.save_preferences(saved)
    assert saved['single']['recordings']==['r1','r2'] and saved['single']['recording']=='r2'
    view.set_recordings(list(reversed(records)))
    assert set(view.rec_tree.selection())=={'r1','r2'} and view.rec_tree.focus()=='r2'
    view.destroy()


def test_mouse_section_switch_and_status_refresh_never_open_files(root, tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from data_analysis_gui.gui import app as gui, widgets
    from data_analysis_gui.gui.canvas import ZoomCanvas
    opened = []
    monkeypatch.setattr(widgets, 'open_path', lambda path: opened.append(path))
    monkeypatch.setattr(ZoomCanvas, 'load', lambda self, path: None)
    monkeypatch.setattr(gui.SingleView, 'check_status', lambda *args: None)
    monkeypatch.setattr(gui, 'load_guide', lambda section: '')
    monkeypatch.setattr(storage, 'load_cached', lambda folder, signature: dict(
        signature=signature, figures=[dict(title='Overview', files={
            ext: f'overview.{ext}' for ext in ('png', 'pdf', 'csv', 'svg')})], tables={}))
    app = SimpleNamespace(prefs={}, status=tk.StringVar(master=root), queued_status=lambda *args: None)
    window = tk.Toplevel(root)
    window.geometry('1200x800')
    view = gui.SingleView(window, app)
    view.pack(fill='both', expand=True)
    record = SimpleNamespace(key='r1', path=Path('run1.xdf'), date='2026-11-02', run='1',
                             data_type='CO8', task_mode='centerout', online=False,
                             has_fpga_clock=True, channel_config=None, label='Record 1')
    try:
        view.set_recordings([record])
        view.rec_tree.selection_set('r1')
        view.rec_tree.focus('r1')
        window.update()
        for sid in ['s01', 's02'] * 4:
            view.status_of[sid] = ('cached', tmp_path, sid, 'r1')
            x, y, width, height = view.sec_tree.bbox(sid)
            view.sec_tree.event_generate('<ButtonPress-1>', x=x+20, y=y+height//2)
            window.update()
            view._set_status((sid, 'r1'), 'cached', tmp_path, sid)
            view.sec_tree.event_generate('<ButtonRelease-1>', x=x+20, y=y+height//2)
            window.update()
            assert view.current_section().id == sid
        assert opened == []
    finally:
        window.destroy()


def test_result_tables_share_one_lazy_pager(root, tmp_path, monkeypatch):
    storage.write_csv(tmp_path / 'first.csv', [dict(value='first')])
    storage.write_csv(tmp_path / 'second.csv', [dict(value='second')])
    reads = []
    read_table = storage.read_table

    def tracked(path):
        reads.append(path.name)
        return read_table(path)

    monkeypatch.setattr(storage, 'read_table', tracked)
    view = ResultView(root)
    view.show(dict(signature='test', figures=[], tables={'First': 'first.csv', 'Second': 'second.csv', 'Missing': 'missing.csv'}),
              tmp_path, guide='说明')
    assert view.summary_text.get().startswith('本次结果：')
    tabs = view.notebook.tabs()
    assert [view.notebook.tab(tab, 'text') for tab in tabs] == ['Summary', 'Table View', '说明']
    pager = view.nametowidget(tabs[1])
    buttons = {w.cget('text'): w for w in walk(pager) if isinstance(w, ttk.Button)}
    assert reads == ['first.csv']
    assert buttons['◀ Prev'].instate(['disabled'])
    first_tree = next(w for w in walk(pager) if isinstance(w, ttk.Treeview))
    buttons['Next ▶'].invoke()
    assert reads == ['first.csv', 'second.csv']
    assert not first_tree.winfo_exists()
    tree = next(w for w in walk(pager) if isinstance(w, ttk.Treeview))
    assert tree.item(tree.get_children()[0], 'values') == ('second',)
    buttons['Next ▶'].invoke()
    assert pager.series_state['index'] == 2
    assert buttons['Next ▶'].instate(['disabled'])
    assert buttons['打开 CSV'].instate(['disabled'])
    assert any('无法读取表格 Missing' in w.get('1.0', 'end') for w in walk(pager) if isinstance(w, tk.Text))
    choice = next(w for w in walk(pager) if isinstance(w, ttk.Combobox))
    choice.current(0)
    choice.event_generate('<<ComboboxSelected>>')
    root.update()
    assert pager.series_state['index'] == 0
    assert buttons['打开 CSV'].instate(['!disabled'])
    view.show(dict(signature='test', figures=[], tables={}), tmp_path)
    assert not pager.winfo_exists()
    assert [view.notebook.tab(tab, 'text') for tab in view.notebook.tabs()] == ['Summary']


def test_single_table_has_disabled_paging_buttons(root, tmp_path):
    storage.write_csv(tmp_path / 'one.csv', [dict(value=1)])
    view = ResultView(root)
    view.show(dict(signature='test', section={'id': 's03'}, figures=[], tables={'Only': 'one.csv'},
                   summary=dict(n_frames=100, n_receive_gaps=2, fpga_clock=False,
                                one_sentence_summary='过时的缓存摘要')), tmp_path)
    assert '100' in view.summary_text.get() and '未知' in view.summary_text.get()
    assert '过时' not in view.summary_text.get()
    pager = view.nametowidget(view.notebook.tabs()[1])
    for button in (w for w in walk(pager) if isinstance(w, ttk.Button) and w.cget('text') != '打开 CSV'):
        assert button.instate(['disabled'])
