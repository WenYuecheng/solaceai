import math
import pytest


def test_billing_rounds_up_and_counts_time_before_training():
    from experiments.qwen3_bonus.budget import RentalBudget
    budget = RentalBudget(hourly_rate=2, cap_yuan=10, started_at=1000, quantum_seconds=60)
    assert budget.cost(1001) == pytest.approx(2 / 60)
    assert budget.cost(4600) == 2
    assert budget.remaining_seconds(1000, reserve_yuan=2) == 14400
    assert not budget.can_start(15400, reserve_yuan=2)


@pytest.mark.parametrize('rate,cap,quantum', [(0,10,60),(-1,10,60),(2,0,60),(2,10,0),(math.nan,10,60),(2,math.inf,60)])
def test_budget_rejects_non_finite_or_invalid_limits(rate,cap,quantum):
    from experiments.qwen3_bonus.budget import RentalBudget
    with pytest.raises(ValueError):
        RentalBudget(hourly_rate=rate, cap_yuan=cap, started_at=0, quantum_seconds=quantum)


def test_future_rental_clock_is_rejected():
    from experiments.qwen3_bonus.budget import RentalBudget
    with pytest.raises(ValueError):
        RentalBudget(2, 10, 1000).cost(999)


def test_budget_deadline_reserves_checkpoint_and_billing_quantum():
    from experiments.qwen3_bonus.budget import RentalBudget
    budget = RentalBudget(3, 10, 0, 60)
    assert budget.deadline(reserve_yuan=1, save_seconds=120) == 10620


def test_reward_preserves_sigmoid_and_repeat_penalty():
    from experiments.qwen3_bonus.rewards import score_response, repetition_ratio
    text = '这件事让你很失落，我们可以先谈谈你最在意的部分。'
    assert score_response(0, text)['score'] == pytest.approx(.5 - .2*repetition_ratio(text))
    assert score_response(10000,text)['score'] <= 1
    repeated = '我理解你的感受。' * 20
    assert score_response(0,repeated)['score'] < score_response(0,text)['score']


@pytest.mark.parametrize('text', ['', '   ', '<tool_call>{}</tool_call>', '你肯定患有抑郁症，应该吃药。'])
def test_invalid_or_clear_diagnostic_output_receives_negative_reward(text):
    from experiments.qwen3_bonus.rewards import score_response
    assert score_response(0,text)['score'] == -1


def test_safety_rule_does_not_punish_safe_negations():
    from experiments.qwen3_bonus.rewards import score_response
    assert score_response(0,'我不能诊断你是否患有抑郁症，可以联系专业人员评估。')['score'] >= 0


def test_reward_nonfinite_is_an_error_not_success():
    from experiments.qwen3_bonus.rewards import score_response
    with pytest.raises(ValueError):
        score_response(math.nan,'我在听。')


def test_calibration_requires_human_review_not_a_model_guess():
    from experiments.qwen3_bonus.rewards import calibration_gate
    rows = [{'id':str(i),'reviewed':False,'expected':'a','score_a':1,'score_b':0} for i in range(20)]
    assert not calibration_gate(rows)['passed']
    for row in rows:
        row.update(reviewed=True, reviewer='human-reviewer')
    assert calibration_gate(rows)['passed']
    rows[0]['id'] = rows[1]['id']
    with pytest.raises(ValueError):
        calibration_gate(rows)
