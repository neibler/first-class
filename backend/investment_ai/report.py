"""Dependency-free standalone HTML research report with SVG equity curves."""

import html
import numpy as np


def equity_chart(histories):
    colors = ["#2563eb", "#059669", "#d97706", "#9333ea"]
    series = {
        key: np.r_[1.0, frame.portfolio_value.to_numpy()]
        for key, frame in histories.items()
    }
    low = min(float(v.min()) for v in series.values())
    high = max(float(v.max()) for v in series.values())
    margin = max((high - low) * 0.1, 0.01)
    low, high = low - margin, high + margin
    svg = [
        '<svg viewBox="0 0 900 360" role="img" aria-label="Final test equity comparison; initial capital 1">'
    ]
    for value in np.linspace(low, high, 5):
        y = 300 - (value - low) / (high - low) * 260
        svg.append(
            f'<path d="M70 {y} H870" stroke="#e2e8f0"/><text x="10" y="{y+5}" font-size="13">{value:.3f}</text>'
        )
    for (name, values), color in zip(series.items(), colors):
        points = " ".join(
            f"{70+i/(len(values)-1)*800:.2f},{300-(value-low)/(high-low)*260:.2f}"
            for i, value in enumerate(values)
        )
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        )
    sample = next(iter(histories.values()))
    svg.append(
        f'<text x="70" y="330" font-size="13">{sample.iloc[0].decision_date}</text><text x="780" y="330" font-size="13">{sample.iloc[-1].date}</text></svg>'
    )
    svg.append(
        "<p>"
        + " · ".join(
            f'<span style="color:{color}">{html.escape(name)}</span>'
            for name, color in zip(series, colors)
        )
        + "</p>"
    )
    return "".join(svg)


def table(rows):
    if not rows:
        return "<p>No rows</p>"

    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.5f}"
        return html.escape(str(value))

    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{html.escape(k)}</th>" for k in rows[0])
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row.values()) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def build_report(output, summary, metrics, histories, validation, stress):
    rows = [{"strategy": key, **value} for key, value in metrics.items()]
    content = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Investment AI 연구 실행 보고서</title><style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc;line-height:1.6}}
section{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:20px 0}}h1,h2{{line-height:1.3}}
.notice{{background:#fff7ed;padding:18px;border-left:4px solid #f97316}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}svg{{width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>Investment AI 연구 실행 보고서</h1><p>실행 ID: <strong>{html.escape(summary['run_id'])}</strong></p>
<div class="notice">{'합성 데이터 기반 기능 검증입니다. 실제 투자 성능의 증거가 아닙니다.' if summary['synthetic'] else '실제 데이터 연구 실행입니다. 데이터의 시점 정확성과 체결 가정은 별도 검증이 필요합니다.'}</div>
<section><h2>실험 설계</h2><p>{summary['dates']}개 날짜 · {summary['assets']}개 ETF · {summary['folds']}개 확장 학습 구간 · {summary['seeds']}개 seed · {summary['candidate_count']}개 보상 후보</p>
<p>보상 선택: <strong>{html.escape(summary['selected_reward'])}</strong>. 검증 구간의 공통 점수 평균으로 선택한 후, 테스트 수익률을 보기 전에 선택을 저장했습니다. 최종 모델은 seed별 목표 비중의 평균을 사용합니다.</p>
<p>최종 테스트: {summary['test_start']} ~ {summary['test_end']}. 초기 자본 1, 거래비용 및 슬리피지 반영. 모든 전략에 동일한 MDD 중단·현금 유지 규칙을 적용합니다.</p></section>
<section><h2>최종 테스트 자산가치</h2>{equity_chart(histories)}{table(rows)}<p>수익률·변동성·MDD는 소수 단위(0.10=10%)입니다. Sharpe는 252거래일 기준 연율화하며 분산이 0이면 표시하지 않습니다. 비용은 초기 자본 대비 누적 차감액입니다. 기간이 짧은 연율화 수익률은 해석에 주의해야 합니다.</p></section>
<section><h2>보상 선택용 Walk-Forward 검증</h2>{table(validation)}<p>각 구간은 자본을 1로 재시작하는 독립 실험입니다. 이 표를 연속 운용 성과로 해석하지 않습니다. 보상 선택에 사용한 검증 성과에는 선택 편향이 있습니다.</p></section>
<section><h2>별도 합성 스트레스 시나리오</h2>{table(stress)}<p>고정된 최종 모델로 별도 seed의 상승·하락·횡보·급락·고변동 시나리오를 실행했습니다. 실제 시장 예측이 아니며 모델 선택에는 사용하지 않습니다.</p></section>
<section><h2>가정과 남은 실데이터 작업</h2><ul><li>t일 종가까지 관측하고 t+1일 종가에 체결합니다. 새 비중은 t+1→t+2 수익률부터 적용합니다.</li><li>리밸런싱은 정해진 거래일 간격이며 체결 가격은 종가로 가정합니다.</li><li>급락으로 15%보다 큰 손실이 발생할 수 있습니다. 중단은 손실 계산 후 작동합니다.</li><li>현재 팀 데이터는 Close 기준으로 배당을 추가 반영하지 않았습니다. 뉴스 적용일은 변경하지 않으며 주문 체결만 1거래일 지연합니다.</li><li>합성 데이터로 최적 보상, 최적 학습량, 실거래 적합성을 결정할 수 없습니다.</li></ul>
<p><a href="comparison.csv">최종 비교 CSV</a> · <a href="validation.csv">검증 CSV</a> · <a href="stress.csv">스트레스 CSV</a> · <a href="latest_weights.csv">최신 목표 비중</a> · <a href="summary.json">실행 요약</a></p></section></html>"""
    (output / "report.html").write_text(content, encoding="utf-8")
