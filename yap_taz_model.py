
# -*- coding: utf-8 -*-
"""
YAP/TAZ - 암세포 동역학 미분방정식 모델

[생물학적 배경]
YAP(Yes-associated protein)/TAZ는 Hippo 신호경로의 핵심 전사공동활성인자로,
인산화되어 세포질에 머무르면 비활성, 탈인산화되어 핵으로 이동하면
TEAD 전사인자와 결합해 세포증식·항세포사멸 유전자 발현을 촉진하는
종양유발(oncogenic) 신호로 작동한다.

- Akt(PI3K-Akt 경로): 비만·고혈당으로 인한 인슐린/IGF-1 신호 과활성화 시
  활성화되며, YAP/TAZ의 핵 내 이동과 안정화를 촉진한다 (Hippo 억제 방향).
- AMPK(AMP-activated protein kinase): 세포 에너지 스트레스(운동으로 인한
  ATP 소모, 에너지 부족)의 센서로 활성화되며, LATS1/2를 경유해
  YAP를 직접 인산화(Ser94)하여 세포질에 가두고 분해를 촉진한다
  (Hippo 활성화 방향, YAP 억제).

[모델 방정식]
dY/dt = alpha*A - beta*M*Y - gamma*Y
dC/dt = r*C*(1 - C/K) + delta*Y*C

Y(t): 활성형(핵 내) YAP/TAZ 농도
C(t): 변이 암세포 수(또는 상대적 종양 크기)
A   : 비만/고혈당 유래 Akt 활성 상수 (YAP 생성·안정화 촉진)
M   : 운동 유래 AMPK 활성 상수 (YAP 인산화·분해 촉진)
"""

import numpy as np
from scipy.integrate import odeint


def yap_taz_odes(state, t, alpha, beta, gamma, r, K, delta, A, M):
    """
    YAP/TAZ - 암세포 동역학 연립미분방정식의 우변(도함수)을 계산한다.

    각 항의 의미:
    ---------------------------------------------------------------
    dY/dt = alpha*A - beta*M*Y - gamma*Y
      + alpha*A   : Akt 신호(A)에 비례해 YAP/TAZ가 새로 생성/안정화되는 항.
                    A가 클수록(비만·고혈당이 심할수록) YAP 농도 증가 속도가 커짐.
      - beta*M*Y  : AMPK 신호(M)가 현재 YAP 농도(Y)에 비례해 YAP를 인산화·
                    분해시키는 항. M과 Y가 모두 클 때(운동을 많이 하고
                    YAP가 많이 쌓여 있을 때) 감소 효과가 가장 크게 나타남
                    (질량작용 법칙, mass-action kinetics).
      - gamma*Y   : 신호와 무관하게 일어나는 YAP의 자연분해(기저 turnover).
                    농도에 비례해서 사라지는 1차 반응.

    dC/dt = r*C*(1 - C/K) + delta*Y*C
      r*C*(1-C/K) : 로지스틱 증식항. 암세포는 초기에 지수적으로 증식하지만
                    자원(혈액공급, 공간)의 한계 K에 가까워질수록 증식이
                    둔화된다. 표준 로지스틱 성장모형과 동일한 구조.
      delta*Y*C   : YAP/TAZ 신호가 암세포 증식을 추가로 가속시키는 항.
                    YAP는 세포증식·항세포사멸 유전자를 전사활성화하므로,
                    YAP 농도(Y)가 높을수록 암세포 수(C)에 비례하여
                    증식 속도가 더 빨라짐 (질량작용 법칙).
    ---------------------------------------------------------------
    """
    Y, C = state
    dYdt = alpha * A - beta * M * Y - gamma * Y
    dCdt = r * C * (1 - C / K) + delta * Y * C
    return [dYdt, dCdt]


def simulate(params, t_max=50, n_points=500, Y0=0.1, C0=1.0):
    """
    주어진 파라미터로 미분방정식을 수치적분(odeint)하여
    시간에 따른 Y(t), C(t) 궤적을 반환한다.

    수치적분을 쓰는 이유: 이 연립미분방정식은 delta*Y*C, beta*M*Y 같은
    비선형(교차)항을 포함해서 손으로 푸는 해석적 해가 존재하지 않는다.
    따라서 아주 작은 시간간격 dt마다 dY/dt, dC/dt를 계산해 Y, C를
    조금씩 갱신해 나가는 수치적 방법(여기서는 LSODA 알고리즘 기반
    odeint)을 사용한다.
    """
    t = np.linspace(0, t_max, n_points)
    state0 = [Y0, C0]
    sol = odeint(
        yap_taz_odes, state0, t,
        args=(params["alpha"], params["beta"], params["gamma"],
              params["r"], params["K"], params["delta"],
              params["A"], params["M"])
    )
    Y = sol[:, 0]
    C = sol[:, 1]
    return t, Y, C


def steady_state_Y(alpha, A, beta, M, gamma):
    """
    YAP 정상상태(dY/dt=0) 농도의 해석해.

    0 = alpha*A - beta*M*Y* - gamma*Y*
    => Y*(beta*M + gamma) = alpha*A
    => Y* = alpha*A / (beta*M + gamma)

    M=0(운동을 전혀 하지 않을 때)이면 Y* = alpha*A/gamma로 최댓값을 갖고,
    M이 커질수록(운동을 많이 할수록) 분모가 커져서 Y*는 단조감소한다.
    """
    return (alpha * A) / (beta * M + gamma)
