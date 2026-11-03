"""Serial worker queue. Every queued item owns an immutable request snapshot.

One process per analysis permits immediate termination of the current section
without discarding later jobs. Worker events carry queue IDs to isolate batches.
"""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from .. import config as C, storage


TERMINAL = {'complete','error','stopped','cleared'}
STATE_LABELS = {'queued':'等待中','running':'运行中','stopping':'正在停止',
                'complete':'已完成','error':'失败','stopped':'已停止','cleared':'已移出队列'}


@dataclass
class Task:
    uid: str
    request: dict
    callback: object
    key: str
    state: str = 'queued'
    detail: str = ''
    worker_task: str = ''
    cached: bool = False

    @property
    def job(self):
        return self.request['jobs'][0]

    @property
    def label(self):
        if self.request.get('report'):
            return '一键报告 · ' + self.request['report']['name']
        return self.job['section'].upper()+' · '+', '.join(Path(p).name for p in self.job['recordings'])


class TaskQueue:
    def __init__(self, events, popen=None):
        self.events = events
        self.popen = popen or subprocess.Popen
        self.items = []
        self.waiting = deque()
        self.current = None
        self.process = None
        self.closing = False

    def _notify(self, task, event):
        if task.callback and not self.closing:
            try:
                task.callback(event)
            except Exception as exc:
                # Display failures must not strand the scheduler or later jobs.
                self.events.put(dict(event='log',line=f'Task callback {task.uid}: {exc}',_queue_id=task.uid))

    def submit(self, request, callback):
        added = []
        existing = {t.key for t in self.items if t.state in ('queued','running')}
        groups = [request['jobs']] if request.get('report') else [[job] for job in request['jobs']]
        for jobs in groups:
            frozen = json.loads(json.dumps(dict(request,jobs=jobs),ensure_ascii=False))
            identity = dict(frozen)
            if identity.get('export') and not identity.get('export_only'):
                # A fresh destination must not defeat compute-job deduplication.
                identity['export'] = dict(layout=identity['export']['layout'])
            key = storage.fingerprint(identity)
            if key in existing:
                continue
            task = Task(f'q{len(self.items)+1}',frozen,callback,key)
            task.worker_task=('report:'+frozen['report']['name'] if frozen.get('report')
                              else task.job['section']+':'+ '+'.join(Path(p).name for p in task.job['recordings']))
            self.items.append(task);self.waiting.append(task);added.append(task)
            existing.add(key)
        return added

    def prioritize(self, uid):
        """Move a waiting task to the front without interrupting the current task."""
        if self.closing:
            return False
        for task in self.waiting:
            if task.uid == uid:
                self.waiting.remove(task)
                self.waiting.appendleft(task)
                return True
        return False

    def start_next(self):
        if self.closing or self.current is not None or not self.waiting:
            return None
        task = self.waiting.popleft()
        self.current = task
        task.state = 'running'
        handle,path = tempfile.mkstemp(prefix='dagui_request_',suffix='.json')
        try:
            with open(handle,'w',encoding='utf-8') as stream:
                json.dump(task.request,stream,ensure_ascii=False)
            flags = subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            process = self.popen([sys.executable,'-u','-X','utf8','-m','data_analysis_gui.worker',path],
                                 cwd=str(C.TOOL_ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',creationflags=flags)
            self.process = process
            threading.Thread(target=self._reader,args=(task.uid,process,path),daemon=True).start()
        except Exception as exc:
            Path(path).unlink(missing_ok=True)
            self.events.put(dict(event='error',_queue_id=task.uid,task=task.worker_task,message=str(exc)))
            self.events.put(dict(event='exit',_queue_id=task.uid,code=-1))
        return task

    def _reader(self, uid, process, path):
        try:
            for line in process.stdout:
                line = line.rstrip('\n')
                event = None
                if line.startswith('@@ '):
                    try:
                        event = json.loads(line[3:])
                    except ValueError:
                        pass
                self.events.put(dict(event or dict(event='log',line=line),_queue_id=uid))
            code = process.wait()
        except Exception as exc:
            self.events.put(dict(event='log',line=str(exc),_queue_id=uid))
            code = -1
        finally:
            Path(path).unlink(missing_ok=True)
        self.events.put(dict(event='exit',code=code,_queue_id=uid))

    def handle(self, event):
        task = self.current
        if task is None or event.get('_queue_id')!=task.uid:
            return None  # late output must never mutate another task or callback
        kind = event['event']
        event['recording_paths'] = list(task.job['recordings'])
        event['queue_id'] = task.uid
        stopping = task.state=='stopping'
        if kind=='plan' and event.get('tasks'):
            task.worker_task = event['tasks'][0]['task']
        elif kind=='message':
            task.detail = event['message']
        elif kind=='complete' and not stopping:
            task.state='complete';task.cached=bool(event.get('cached'))
            task.detail='已读取存储结果' if task.cached else '已计算并存储'
        elif kind=='error' and not stopping:
            task.state='error';task.detail=event.get('message','失败')
        elif kind=='exit':
            if stopping:
                task.state='stopped';task.detail='已终止当前进程；其余等待任务继续'
                event['cancelled']=True
            elif task.state not in TERMINAL:
                task.state='error';task.detail=f'计算进程未返回结果（退出码 {event["code"]}）'
                self._notify(task,dict(event='error',task=task.worker_task or task.label,
                                      message=task.detail,recording_paths=event['recording_paths'],queue_id=task.uid))
            self.current=None;self.process=None
        if task.callback and kind not in ('log','done') and (not stopping or kind=='exit'):
            self._notify(task,event)
        return task

    def stop_current(self):
        task = self.current
        if task is None or task.state in TERMINAL or task.state=='stopping':
            return False
        task.state='stopping'
        if self.process and self.process.poll() is None:
            self.process.terminate()
            process = self.process
            def force_stop():
                if process.poll() is None:
                    process.kill()
            timer = threading.Timer(2,force_stop)
            timer.daemon=True
            timer.start()
        return True

    def clear_waiting(self):
        removed = list(self.waiting)
        self.waiting.clear()
        for task in removed:
            task.state='cleared';task.detail='等待任务已移出队列'
            self._notify(task,dict(event='cancelled',queue_id=task.uid,recording_paths=task.job['recordings']))
        return removed

    def close(self):
        self.closing=True
        self.clear_waiting()
        self.stop_current()
