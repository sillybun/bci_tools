"""Queue snapshot isolation, event routing, stopping and continuation in real processes."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

from data_analysis_gui.gui.task_queue import TaskQueue


def request(*sections, **params):
    return dict(criteria={'window_start':.2},output_root='out',data_root='data',
                jobs=[dict(section=s,version='v1',recordings=[f'{s}.xdf'],params=dict(params)) for s in sections])


def test_snapshots_duplicate_detection_and_clear_waiting():
    events=queue.Queue();callbacks=[]
    manager=TaskQueue(events)
    source=request('s01','s02',alpha=.05)
    added=manager.submit(source,callbacks.append)
    source['criteria']['window_start']=99
    source['jobs'][0]['params']['alpha']=.8
    assert added[0].request['criteria']['window_start']==.2
    assert added[0].job['params']['alpha']==.05
    assert not manager.submit(request('s01',alpha=.05),callbacks.append)
    assert len(manager.submit(request('s01',alpha=.01),callbacks.append))==1
    removed=manager.clear_waiting()
    assert len(removed)==3 and all(t.state=='cleared' for t in removed)
    assert len(callbacks)==3 and all(e['event']=='cancelled' for e in callbacks)
    assert len(manager.submit(request('s01',alpha=.05),callbacks.append))==1


def test_late_events_and_unexpected_exit_route_to_the_right_callback():
    events=queue.Queue();first=[];second=[]
    manager=TaskQueue(events)
    a=manager.submit(request('s01'),first.append)[0]
    b=manager.submit(request('s02'),second.append)[0]
    manager.waiting.popleft();manager.current=a;a.state='running'
    manager.handle(dict(event='exit',_queue_id=a.uid,code=7))
    assert a.state=='error' and first[0]['event']=='error'
    assert first[0]['recording_paths']==['s01.xdf']
    manager.waiting.popleft();manager.current=b;b.state='running'
    assert manager.handle(dict(event='complete',_queue_id=a.uid,task='s01:s01.xdf')) is None
    assert b.state=='running' and not second
    manager.handle(dict(event='complete',_queue_id=b.uid,task='s02:s02.xdf',cached=True))
    manager.handle(dict(event='exit',_queue_id=b.uid,code=0))
    assert b.state=='complete' and b.cached and second[0]['recording_paths']==['s02.xdf']


def test_prioritize_moves_only_the_selected_waiting_task():
    manager = TaskQueue(queue.Queue())
    current, first, second, selected = manager.submit(request('s01', 's02', 's03', 's04'), None)
    manager.current = manager.waiting.popleft()
    current.state = 'running'
    assert manager.prioritize(selected.uid)
    assert manager.current is current
    assert list(manager.waiting) == [selected, first, second]
    assert not manager.prioritize(current.uid)
    assert list(manager.waiting) == [selected, first, second]


CHILD = '''
import json,sys,time
r=json.load(open(sys.argv[1],encoding='utf-8'));j=r['jobs'][0];task=j['section']+':child'
def emit(event,**values):print('@@ '+json.dumps(dict(event=event,**values)),flush=True)
emit('plan',tasks=[dict(task=task)])
emit('start',task=task,label=task)
emit('message',task=task,message='ready')
if j['params'].get('hang'):time.sleep(30)
emit('complete',task=task,cached=False)
emit('done',failures=0)
'''


def test_real_worker_stop_only_current_then_run_newly_queued_job():
    events=queue.Queue();calls=[];processes=[];paths=[]
    def launch(args,**kw):
        paths.append(Path(args[-1]))
        child=subprocess.Popen([sys.executable,'-u','-X','utf8','-c',CHILD,args[-1]],**kw)
        processes.append(child)
        return child
    manager=TaskQueue(events,popen=launch)
    a=manager.submit(request('s01',hang=True),calls.append)[0]
    manager.start_next()
    try:
        while True:
            event=events.get(timeout=5);manager.handle(event)
            if event['event']=='message':break
        b=manager.submit(request('s02'),calls.append)[0]
        assert manager.current==a and list(manager.waiting)==[b]
        assert manager.stop_current()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            event=events.get(timeout=5);manager.handle(event)
            manager.start_next()
            if manager.current is None and not manager.waiting:break
        assert a.state=='stopped' and b.state=='complete'
        assert all(p.poll() is not None for p in processes)
        assert any(e['event']=='exit' and e.get('cancelled') and e['queue_id']==a.uid for e in calls)
        assert not any(e['event']=='complete' and e['queue_id']==a.uid for e in calls)
        assert any(e['event']=='complete' and e['queue_id']==b.uid for e in calls)
        assert all(not path.exists() for path in paths)
    finally:
        manager.close()
        for p in processes:
            if p.poll() is None:p.kill()
            p.wait(timeout=5)


def test_close_discards_waiting_and_never_launches_more():
    manager=TaskQueue(queue.Queue())
    tasks=manager.submit(request('s01','s02'),lambda e:None)
    manager.close()
    assert all(t.state=='cleared' for t in tasks)
    assert manager.start_next() is None
