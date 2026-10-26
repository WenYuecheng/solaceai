import pytest


def test_resume_rejects_changed_or_duplicate_capture():
    from experiments.qwen3_bonus.perm_capture import validate_existing
    case={'id':'one','prompt':[{'role':'user','content':'fixed'}]}
    row={'id':'one','variant':'base','prompt':case['prompt'],'response':'actual',
         'base_revision':'revision','converted_base_sha256':'base',
         'chat_template_sha256':'template','adapter_sha256':None,'temperature':0,'max_tokens':256,'quantization':'4bit'}
    identity={k:row[k] for k in ['base_revision','converted_base_sha256','chat_template_sha256']}
    assert validate_existing([row],[case],{'base':None},identity)=={('one','base')}
    for changed in [[row,row],[{**row,'prompt':[]}],[{**row,'adapter_sha256':'changed'}],[{**row,'max_tokens':512}]]:
        with pytest.raises(ValueError):validate_existing(changed,[case],{'base':None},identity)


def test_test_models_require_frozen_dev_selection(tmp_path):
    from experiments.qwen3_bonus.perm_capture import select_jobs
    with pytest.raises(ValueError,match='selection'):
        select_jobs('test',tmp_path,None)
