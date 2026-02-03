from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ALBreakoutPullback50pctParams:
    break_out_series_n: int = 3
    break_out_n_bars: int = 10
    BO_n_times_atr: float = 1.0
    n_bar_pivot: int = 3  # pivot高低點定義K線數
    pullback_ratio: float = 0.5  # 回調百分比
    max_notional_pct: float = 1.0
    min_qty: float = 0.001
    sl_atr_like: float = 0.0  # MVP不做ATR，示範保留欄位
    fixed_sl_pct: float = 0.01
    rr: float = 2.0           # TP = SL距離 * rr
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL
class ALBreakoutPullback50pctStrategy(Strategy):
    """
    Docstring for ALBreakoutPullback50pctStrategy
    
    策略邏輯：
    1. 偵測breakout
    2. breakout後等待PB pullback_ratio 幅度
    3. 如果PB K線碰到pullback_ratio 幅度(多頭為例)，則在該位置下單進場
    4. 停損設在channel下方(多頭為例)，停利設在停損距離的RR倍
    5. 移動停損() -- todo
    """

    def __init__(self, params: ALBreakoutPullback50pctParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.bo_start_i = None  # 紀錄breakout開始的index
        self.pb_start_i = None  # 紀錄pullback開始的index
        # self.pb_bar_define_n = 2  # pullback定義:連續ll K線數
        # self.tp_changed = False  # 是否已經移動過停利

    def required_indicators(self) -> Dict[str, Any]:
        n = self.p.break_out_series_n
        return {
            "rocp_1": ("rocp", 1),
            f"rocp_{n}": ("rocp", n),
            "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
            "ll": ("rolling_low", self.p.break_out_n_bars, "low"),
            "atr": ("atr", 14),
            "ma": ("ma", 20, "close", "EMA"),
            "streak": ("bar_streak", "ignore"),
            "ll_streak": ("hh_ll_streak", "ll"),
            "hh_streak": ("hh_ll_streak", "hh"),
            "pivot_low_mask": ("pivot_mask", "low", self.p.n_bar_pivot),
        }

    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        i = ctx.i
        df = ctx.df
        pos = ctx.position
        init_equity = ctx.init_equity
        now_equity = ctx.now_equity
        if self.p.sizing_equity_base == SizingEquityBase.INITIAL:
            base_equity = init_equity
        elif self.p.sizing_equity_base == SizingEquityBase.CURRENT:
            base_equity = now_equity
        intents: List[OrderIntent] = []

        open_series = df["open"]
        close_series = df["close"]
        low_series = df["low"]
        high_series = df["high"]

        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:
            # if pos.side == Side.LONG:
            #     if (self.state == "BO" or self.state == "UPDATE_SL"):
            #         # 檢查是否PB
            #         ll_streak = ctx.indicators["ll_streak"]
            #         if ll_streak.iat[i] >= self.pb_bar_define_n:
            #             self.state = "PB"
            #             self.pb_start_i = i
            #     elif self.state == "PB":
            #         # 檢查是否REUP
            #         ## 比前高高
            #         last_hh = float(high_series.iat[self.pb_start_i - self.pb_bar_define_n]) if (self.pb_start_i - self.pb_bar_define_n) >=0 else float("nan")
            #         cond_reup_higher = high_series.iat[i] > last_hh
            #         if cond_reup_higher:
            #             self.state = "REUP"
            #     elif self.state == "REUP":
            #         # 檢查有沒有pivot low
            #         start_i = self.pb_start_i if self.pb_start_i is not None else 0
            #         end_i = i - self.p.n_bar_pivot  # 只允許用到已確認 pivot（避免 look-ahead）
            #         if end_i >= start_i:
            #             mask = ctx.indicators["pivot_low_mask"].iloc[start_i:end_i+1].to_numpy()
            #             if mask.any():
            #                 rel = np.flatnonzero(mask)[-1]
            #                 lastest_pivot_low_i = start_i + int(rel)
            #             else:
            #                 lastest_pivot_low_i = None
            #         else:
            #             lastest_pivot_low_i = None
            #         lastest_pivot_low = float(low_series.iat[lastest_pivot_low_i]) if lastest_pivot_low_i is not None else float("nan")
            #         if not np.isnan(lastest_pivot_low):
            #             # 找到pivot low，更新停損到pivot low
            #             new_sl_price = lastest_pivot_low if lastest_pivot_low > pos.sl_price else pos.sl_price
            #             # 更新到leg1.2 MM的rr倍
            #             origin_sl_range = pos.avg_price - pos.sl_price if pos.sl_price is not None else 0.0
            #             new_tp_price = pos.tp_price if self.tp_changed else new_sl_price + origin_sl_range * self.p.rr
            #             self.tp_changed = True
            #             intents.append(
            #                 OrderIntent(
            #                     action=ActionType.UPDATE,
            #                     side=pos.side, # side不會用到
            #                     qty=0.0, # qty不會用到
            #                     sl_price=new_sl_price,
            #                     tp_price=new_tp_price,
            #                     be_price=None,
            #                     priority=50, 
            #                 )
            #             )
            #             self.state = "UPDATE_SL"
            # debug_info = {
            #     "state": self.state,
            # }
            return intents

        bar_streak = ctx.indicators["streak"]
        streak_prev = int(bar_streak.iat[i - 1]) if i - 1 >= 0 else 0
        streak_prev2 = int(bar_streak.iat[i-2]) if i - 2 >= 0 else 0
        streak_curr = int(bar_streak.iat[i])

        if i < 4:
            return intents
        # i為進場K線，i-1為反向K線R1，i-2為突破的最後一根K線



        
        # 無倉，檢查進場
        else:
            # 沒有倉位時，檢查狀態
            ## 1. 一開始默認state = None，檢查有沒有 micro channel or breakout


            

            
            hh = ctx.indicators["hh"]
            # 前streak_prev根開盤到收盤的幅度(BO range)
            


            # 檢查前兩根是否為多頭突破

            ## 連續n根同向K線以上(BO)
            cond1 = streak_curr >= self.p.break_out_series_n
            ## 突破前高(BO、MC)
            cond2 = high_series.iat[i] > hh.iat[i - 1]
            
            ## 收盤大於MA(BO、MC)
            cond4 = close_series.iat[i] > ctx.indicators["ma"].iat[i]
            ## 連續4根K線沒有PB(MC)
            cond5 = sum([ctx.indicators["ll_streak"].iat[j] for j in range(i - 4, i)]) == 0
            ## 策略條件做多或雙向
            cond6 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)

            if (cond1 and cond2 and cond4 and cond6):
                # 記錄breakout開始index
                ## 突破幅度>=atr最小幅度(BO、MC)
                bo_open_p = float(open_series.iat[i - streak_curr + 1]) if i - streak_curr + 1 >= 0 else 0.0
                bo_range = abs(close_series.iat[i] - bo_open_p) if streak_curr > 0 else 0.0
                cond3 = bo_range >= self.p.BO_n_times_atr * float(ctx.indicators["atr"].iat[i])
                if cond3:
                    self.state = "BO"
                    self.bo_start_i = i - streak_curr + 1

            elif (cond2 and cond4 and cond5 and cond6):
                # 記錄breakout開始index
                ## 突破幅度>=atr最小幅度(BO、MC)
                bo_open_p = float(open_series.iat[i - 3])
                bo_range = abs(close_series.iat[i] - bo_open_p)
                cond3 = bo_range >= self.p.BO_n_times_atr * float(ctx.indicators["atr"].iat[i])
                if cond3:
                    self.state = "MC"
                    self.bo_start_i = i - 3


            # 狀態已經是BO或MC，檢查PB起點
            if self.state in ("BO", "MC"):
                ## 出現兩根ll K線以上
                pb_cond1 = ctx.indicators["ll_streak"].iat[i] >= 2
                if pb_cond1:
                    self.state = "PB"
                    self.pb_start_i = i

            # 狀態已經是PB，檢查進場點
            elif self.state == "PB":
                # 計算bo 高低點 pullback ratio 位置
                bo_high = float(high_series.iloc[self.bo_start_i:self.pb_start_i+1].max())
                bo_low  = float(low_series.iloc[self.bo_start_i:self.pb_start_i+1].min())
                pb_level = bo_high - (bo_high - bo_low) * self.p.pullback_ratio
                # 檢查有沒有碰到pb_level
                long_cond_entry = low_series.iat[i] <= pb_level <= high_series.iat[i]
                if long_cond_entry:
                    # 進場價格設在pb_level buy limit
                    entry_price = pb_level
                    # 停損設在channel下方
                    sl_price = bo_low
                    sl_distance = entry_price - sl_price
                    # 停利設在停損距離的RR倍
                    tp_price = entry_price + sl_distance * self.p.rr

                    # 計算可用資金與下單數量
                    max_notional_lose = base_equity * (self.p.max_notional_pct / 100)
                    qty = max_notional_lose / (abs(entry_price - sl_price)) if abs(entry_price - sl_price) > 0 else 0.0
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.LONG,
                            qty=max(self.p.min_qty, float(qty)),
                            tp_price=tp_price,
                            sl_price=sl_price,
                            be_price=None,
                            priority=10,
                        )
                    )
                    self.state = None


        debug_info = {
            "cond1": cond1,
            "cond2": cond2,
            # "cond3": cond3,
            "cond4": cond4,
            "cond5": cond5,
            "cond6": cond6,
            "state": self.state,
        }
        return intents
