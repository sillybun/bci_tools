import hashlib
from PIL import Image
from data_analysis_gui import config as C,storage
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.sections.base import SectionResult
from data_analysis_gui.sections.x01_channel_stability_v1.section import SECTION
from data_analysis_gui.sections.x01_channel_stability_v1.chip_layout import append_chip_layout,TITLE


def test_reference_export_preserves_original_and_asset_changes_invalidate_cache(tmp_path,monkeypatch):
    source=tmp_path/'BGA_shank_projection.png'
    Image.new('RGB',(60,40),'red').save(source)
    monkeypatch.setattr(C,'BGA_SHANK_PROJECTION',source)
    def inputs():
        return storage.signature_inputs(SECTION,[],TrialFilter(),SECTION.defaults())
    before=inputs()
    result=SectionResult();append_chip_layout(result)
    assert result.figures[0][0]==TITLE
    folder=tmp_path/'result'
    record=storage.save_result(result,folder,storage.fingerprint(before),before,SECTION,[])
    figure=record['figures'][0]
    assert (folder/figure['files']['png']).read_bytes()==source.read_bytes()
    assert (folder/figure['files']['csv']).stat().st_size<10000
    assert all((folder/figure['files'][ext]).is_file() for ext in ('pdf','svg'))
    assert result.summary['chip_layout_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    Image.new('RGB',(60,40),'blue').save(source)
    assert storage.fingerprint(inputs())!=storage.fingerprint(before)


def test_missing_reference_is_visible_in_notes(tmp_path):
    result=SectionResult();append_chip_layout(result,tmp_path/'missing.png')
    assert not result.figures
    assert '参考图缺失' in result.notes[0]
