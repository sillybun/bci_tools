"""Produce a separate GUI-ready package from selected multi-session runs.

uv run python scripts/split_session_package.py --source-package PATH --output PATH \
    --run 1109/CO8_run-006 --run 1111/GRID_run-001
"""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.channels import read_channel_config
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.xdf_split import digest, split_recording
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--resume-stage', type=Path)
    args = parser.parse_args()
    package, destination = args.source_package.resolve(), args.output.resolve()
    if destination.exists() or destination.is_relative_to(package):
        raise ValueError('Output must be a new directory outside the original package')
    sources = []
    for relative in args.run:
        run = (package / relative).resolve()
        if not run.is_relative_to(package) or not run.is_dir():
            raise ValueError(f'Invalid source run: {relative}')
        files = list((run / 'xdf').glob('*.xdf'))
        if len(files) != 1:
            raise ValueError(f'Expected exactly one XDF in {run}')
        dates = {re.search(r'_(\d{8})-\d{6}$', p.name)[1] for p in (run / 'sessions').iterdir()}
        if len(dates) != 1:
            raise ValueError('Run has ambiguous recording dates')
        sources.append((run, files[0], next(iter(dates))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_stage:
        stage = args.resume_stage.resolve()
        if (stage.parent != destination.parent or not stage.name.startswith(destination.name + '.partial-')
                or not stage.is_dir()):
            raise ValueError('Resume directory is outside the expected staging location')
    else:
        stage = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=destination.parent))
    copies, all_rows = [], []

    def copy_verified(src, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        sha = digest(src)
        if digest(target) != sha:
            raise ValueError(f'Copy hash mismatch: {src}')
        copies.append(dict(source=str(src), output=target.relative_to(stage).as_posix(), sha256=sha))

    for run, source, date in sources:
        target_run = stage / date / run.name
        if args.resume_stage and (target_run / 'sessions').exists():
            rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
                (target_run / 'sessions').glob('*/xdf_split_manifest.json'))]
            if len(rows) != len(list((run / 'sessions').iterdir())):
                raise ValueError('Incomplete split stage; cannot resume')
            for row in rows:
                path = Path(row['output'])
                if not path.is_absolute():
                    path = stage / path
                if digest(source) != row['source_sha256'] or digest(path) != row['output_sha256']:
                    raise ValueError('Resume source/output hash mismatch')
        else:
            rows = split_recording(source, target_run / 'sessions')
        for row in rows:
            original_session = run / 'sessions' / row['session']
            target_session = target_run / 'sessions' / row['session']
            for file in original_session.rglob('*'):
                if not file.is_file():
                    continue
                relative = file.relative_to(original_session)
                # Current GUI's refit directory contains the actual session files.
                if len(relative.parts) > 2 and relative.parts[0] == 'refit' and re.fullmatch(
                        r'\d{8}_\d{6}', relative.parts[1]):
                    relative = Path('refit', *relative.parts[2:])
                copy_verified(file, target_session / relative)
            initial = [p for p in (target_session / 'initial_weights').rglob('*.pkl')
                       if p.name != 'INITIAL_position_velocity_kf.pkl']
            if len(initial) == 1:
                copy_verified(initial[0], target_session / 'initial_weights' / 'INITIAL_position_velocity_kf.pkl')
                copies[-1]['source'] = str(next((original_session / 'initial_weights').rglob('*.pkl')))
            path = Path(row['output'])
            row['output'] = path.relative_to(stage).as_posix() if path.is_absolute() else path.as_posix()
            path = stage / row['output']
            recording = describe(path, stage)
            if recording.reason:
                raise ValueError(recording.reason)
            session = Session.load(recording)
            trial_csv = target_session / 'behavior' / 'trials.csv'
            with trial_csv.open(encoding='utf-8-sig', newline='') as handle:
                csv_trials = sum(1 for _ in csv.DictReader(handle))
            # With the 0.5 s context, all task markers must survive Session.load.
            expected_ends = sum(
                int(r['samples']) for r in row['streams'].values() if r['name'] == 'cursortasks_events')
            row['gui_validation'] = dict(date=recording.date, task_mode=recording.task_mode,
                counts=list(session.counts.shape), reconstructed_trials=len(session.trials),
                complete_trials=sum(t['complete'] for t in session.trials),
                trial_end_events=sum(e[0] == 'T_TRIAL_END' for e in session.events),
                raw_event_samples_including_duplicate_streams=expected_ends,
                full_session_behavior_csv_rows=csv_trials)
            del session
            row['coverage_note'] = (
                'The XDF contains only this source recording segment. Copied behavior CSV and refit '
                'artifacts describe the complete task session and may cover more time. No missing '
                'XDF trials were fabricated; short startup segments are retained.')
            (target_session / 'xdf_split_manifest.json').write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            all_rows.append(row)
            print(row['session'], row['gui_validation'], flush=True)
        log_dir = run.parent / 'shared_logs' / 'acquisition' / 'neuralinsight'
        for file in log_dir.glob('*.log'):
            target = stage / date / 'shared_logs' / 'acquisition' / 'neuralinsight' / file.name
            if not target.exists():
                copy_verified(file, target)
    for row in all_rows:
        path = stage / row['output']
        recording = describe(path, stage)
        read_channel_config(recording.channel_config)
        model_files = manifest(recording, {})
        row['gui_validation']['saved_model_weights'] = len(model_files['weights']) if model_files else 0
        (path.parent.parent / 'xdf_split_manifest.json').write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(source_package=str(package), original_xdf_files_unchanged=True,
                  source_run_count=len(sources), split_session_count=len(all_rows), sessions=all_rows,
                  copied_sidecars=copies,
                  policy='Task streams by session_name; shared streams within task coverage +/- 0.5 s; '
                         'original duplicate streams preserved; no timestamp rebasing or resampling; '
                         'all retained samples and timestamps verified exactly with pyxdf')
    (stage / 'split_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (stage / 'README.txt').write_text(
        '按任务 session 拆分的 XDF 派生包\n\n'
        '原整理包及原 XDF 保留不变。本目录在 GUI 中作为独立数据根目录使用，避免与原文件重复统计。\n'
        '神经与 decoder 流按本 session 的实际行为流覆盖范围前后各保留 0.5 秒；保留同 session 的全部行为样本与重复流。\n'
        '时间戳不归零、不同步、不去抖、不重采样。逐流样本、时间戳、clock offset 均与来源逐项核对。\n'
        'session 间空闲时段不收入派生文件，仍完整保存在原始 XDF 中。\n'
        '配套 config/log/behavior/初始权重/refit 已复制并校验 SHA-256；refit 的时间子目录在派生包中提升一层以供 GUI 定位。\n'
        '行为 CSV 与 refit 可能覆盖完整任务，会比当前 XDF 片段长。1111 GRID 的 11:31:17 文件只是下一 8×8 会话开头，'
        '不等于原 GRID_run-002 的完整录制。1109 的 14:09:11 是 12×12 启动片段，0 个结束试次，仍保留。\n'
        'split_manifest.json 和各 session/xdf_split_manifest.json 记录来源哈希、裁剪边界、流样本数及 GUI 核验。\n'
        '格式依据：https://github.com/sccn/xdf/wiki/Specifications\n', encoding='utf-8')
    stage.rename(destination)
    print('VERIFIED_OUTPUT', destination, flush=True)


if __name__ == '__main__':
    main()
