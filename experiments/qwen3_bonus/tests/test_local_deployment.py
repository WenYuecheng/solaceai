import json
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.config import Settings
from app.schemas.dtos import AiMessage
from experiments.qwen3_bonus import integration


def test_replacement_injects_classifier_without_replacing_safety(monkeypatch):
    import app.agents.graph_runtime as graph
    seen = {}
    def init(self, db, settings, **kwargs):
        seen.update(kwargs)
        self.settings = settings
        self.client = kwargs['decision_client']
    monkeypatch.setattr(graph.GraphAgentRuntime, '__init__', init)
    local = Mock()
    emotion = Mock()
    local.for_model.return_value = emotion
    integration.make_runtime_factory(local, replace_emotion=True)(None, Settings(_env_file=None))
    assert seen['emotion_client'] is emotion
    assert 'safety_client' not in seen and 'intent_client' not in seen


def test_local_assessment_uses_training_prompt_and_correct_metadata(monkeypatch):
    import app.agents.graph_runtime as graph
    from apps.offline.assessment import OfflineAssessmentService
    local = Mock()
    emotion = Mock()
    emotion.complete.return_value = '焦虑'
    local.for_model.return_value = emotion
    factory = integration.make_runtime_factory(local, replace_emotion=True)
    settings = Settings(_env_file=None, risk_assessment_source='local', ai_provider='openai')
    def existing(settings, safety, text, history=None, emotion_client=None):
        return OfflineAssessmentService(settings, emotion_client).assess(text, history)
    monkeypatch.setattr(integration, 'original_assessment', existing)
    result, observation = factory.assess(settings, None, '我的电话13812345678，我担心考试')
    assert observation['provider'] == 'vllm-metal'
    assert observation['model'] == 'solace-emotion'
    assert observation['status'] == 'ok'
    prompt = emotion.complete.call_args.args[0]
    assert '13812345678' not in prompt[-1].content
    assert '分析用户文本情绪，只能输出：正常、焦虑、低落、高风险' in prompt[-1].content


def test_graph_payload_is_projected_to_real_dialogue_and_evidence():
    messages = [AiMessage(role='system', content='原生工具调用说明'), AiMessage(role='user', content=json.dumps({
        'task': '我不想听建议，只想说说', 'history': [{'role':'assistant','content':'最近什么让你难受？'}],
        'loaded_skills': {'listen':'先倾听'}, 'evidence':[{'chunk_id':17,'content':'倾听资料'}],
        'tools':['write_memory'], 'high_risk':False, 'memory':'前次考试压力', 'observations':[],
    },ensure_ascii=False))]
    projected = integration.local_response_messages(messages)
    assert projected[-1].content == '我不想听建议，只想说说'
    assert projected[-2].content == '最近什么让你难受？'
    combined = '\n'.join(m.content for m in projected)
    assert '先倾听' in combined and '倾听资料' in combined and '17' in combined
    assert 'write_memory' not in combined and '原生工具调用说明' not in combined


def test_candidate_activation_requires_matching_real_smoke(tmp_path, monkeypatch):
    import pytest
    from experiments.qwen3_bonus.local_app import validate_candidate
    from experiments.qwen3_bonus import asgi
    config = SimpleNamespace()
    monkeypatch.setattr(asgi, 'deployment_fingerprints', lambda cfg: {'checked':'weights'})
    report = tmp_path/'acceptance.json'
    report.write_text(json.dumps({'mode':'user_authorized_local_candidate', 'deployment_artifacts':{'checked':'weights'},
                                 'genuine_inference':False,'smoke_passed':True}))
    with pytest.raises(ValueError):validate_candidate(report, config)
    data=json.loads(report.read_text());data.update(genuine_inference=True,classification_cases=4,alternating_requests=6)
    report.write_text(json.dumps(data));assert validate_candidate(report,config)['genuine_inference']
    data['deployment_artifacts']={'changed':'weights'};report.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_candidate(report, config)


def test_m4_memory_limits_and_private_engine_logs(tmp_path):
    from experiments.qwen3_bonus.serving import ServingConfig,engine_argv
    cfg=ServingConfig(model_path='/tmp/base',emotion_adapter='/tmp/emotion',response_adapter='/tmp/response')
    args=engine_argv(cfg)
    assert args[args.index('--gpu-memory-utilization')+1]=='0.45'
    assert args[args.index('--max-num-batched-tokens')+1]=='512'
    assert '--no-enable-log-requests' in args
