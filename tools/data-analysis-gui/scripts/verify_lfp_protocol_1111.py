"""Cross-check every historical LFP packet with the user-supplied reader.

Dependency (isolated from project):
uv pip install --target .local/lfp_protocol_deps protobuf==6.33.4
"""
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=Path('E:/AitBCI/data/LFP')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.local/lfp_protocol_deps'))
sys.path.insert(0,str(PROTOCOL))
from read_lfp import read_lfp
from implant_decoder_pb2 import ImplantDecoderFrame
from data_analysis_gui.core.xdf_io import load_streams

SOURCE=Path('E:/AitBCI/data/keaiduo/20261111/run-009-CO8-offline-LFP')
OUT=ROOT/'output/lfp_20261111_run009'


def main():
    streams,_=load_streams(next((SOURCE/'xdf').glob('*.xdf')))
    prior=np.load(OUT/'unpacked_lfp.npz')
    existing=prior['values'];ids=prior['channel_ids'].tolist()
    values=[];present=Counter();ticks=Counter();moduli=Counter();align=Counter()
    scalar=['schema_version','sequence','sdk_sequence','host_monotonic_ns','device_timestamp_100us',
            'sample_format','samples_per_channel','sdk_dropped_batches','pipe_dropped_frames',
            'sdk_dropped_events','host_dropped_events','channel_mapping_generation','profile_id','profile_sha256','mode']
    previous_sequence=None
    for row in streams['lfp_packets']['time_series']:
        p=json.loads(row[0]);payload=base64.b64decode(p['payload_base64'],validate=True)
        assert len(payload)==p['payload_bytes'] and len(payload)+4==p['framed_bytes']
        # XDF omits the pipe framing header; restore it only for the supplied pipe reader.
        packet=io.BytesIO(struct.pack('<I',len(payload))+payload)
        frame,by_channel=read_lfp(packet)
        assert packet.read()==b''
        assert frame.mode==ImplantDecoderFrame.LFP and frame.sample_format==ImplantDecoderFrame.INT16_LE
        for name in scalar:
            assert getattr(frame,name)==p[name],(name,getattr(frame,name),p[name])
        assert list(frame.channels)==ids==p['wire_channel_ids']
        assert frame.samples_per_channel==20 and len(frame.samples)==240
        assert previous_sequence is None or frame.sequence==previous_sequence+1
        previous_sequence=frame.sequence
        values.append(np.array([by_channel[c] for c in ids],dtype=np.int16).T)
        present.update(f.name for f,_ in frame.ListFields())
        ticks[frame.device_timestamp_tick_us]+=1
        moduli[frame.device_timestamp_modulus]+=1
        if frame.HasField('time_alignment'):
            align[frame.time_alignment.Status.Name(frame.time_alignment.status)]+=1
        else:
            align['absent']+=1
    decoded=np.concatenate(values)
    np.testing.assert_array_equal(decoded,existing)
    report=dict(protocol_files={p.name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in PROTOCOL.glob('*.py')},
        reader='User-supplied read_lfp.read_lfp, using generated ImplantDecoderFrame with protobuf 6.33.4',
        packets=len(values),shape=list(decoded.shape),scalar_metadata_matches=True,
        exact_prior_sample_match=True,compared_scalar_samples=int(decoded.size),maximum_absolute_sample_difference=0,
        mode=dict(value=ImplantDecoderFrame.LFP,name='LFP'),sample_format=dict(value=ImplantDecoderFrame.INT16_LE,name='INT16_LE'),
        layout='channel-major',channel_numbering='zero-based physical channel IDs 0..1023, as documented in read_lfp.py',
        zero_based_physical_channel_ids=ids,one_based_display_ids_only=[i+1 for i in ids],
        populated_field_packet_counts=dict(present),device_timestamp_tick_us_values=dict(ticks),
        device_timestamp_modulus_values=dict(moduli),time_alignment_packet_counts=dict(align),
        missing_meanings=['ADC count to microvolt scaling','frontend filters and downsampling details','physical sampling rate',
                          'whether packet device timestamp refers to first or last sample','absolute device-to-task acquisition latency'],
        interpretation='Protocol confirms prior samples and channel order exactly. Zero defaults in absent proto3 scalar fields are unspecified, not a zero-duration tick. Old timing estimation and decoding results remain unchanged; no re-training is needed for this protocol verification.')
    (OUT/'protocol_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='protocol_files'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
