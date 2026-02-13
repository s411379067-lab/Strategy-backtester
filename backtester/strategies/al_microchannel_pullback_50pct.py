from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ALBreakoutPullback50pctParams:
    """
    Parameters for ALBreakoutPullback50pctStrategy
    
    Attributes:
        break_out_series_n (int): 突破連續K線數
        break_out_n_bars (int): 突破區間長度（K線數）
        BO_n_times_atr (float): 突破幅度需大於ATR倍數
        n_bar_pivot (int): pivot高低點定義K線數
        pullback_ratio (List[float]): 回調百分比(多空可用不同值)
        min_sl_range_pct (float): 最小停損距離百分比
        max_notional_pct (float): 最大虧損佔資金比例%
        min_qty (float): 最小下單數量
        sl_atr_like (float): 停損距離ATR倍數(此策略不使用)
        fixed_sl_pct (float): 固定停損百分比(此策略不使用)
        rr (float): 風險報酬比
        time_exit_bars (int): 時間出場K線數
        allow_side (Optional[Side]): 允許進出場方向
    """
    break_out_series_n: int = 3
    break_out_n_bars: int = 10
    BO_n_times_atr: float = 1.0
    n_bar_pivot: int = 3  # pivot高低點定義K線數
    pullback_ratio: List[float] = field(default_factory=lambda: [0.5])  # 回調百分比
    min_sl_range_pct: float = 0.3  # 最小停損距離百分比
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
    1. BO定義: 1.連續n根同向K線以上。2.且突破前高。3.收盤在MA之上。4.實體越來越強。5.策略條件允許的方向。6.突破幅度大於ATR倍數且大於固定百分比。
    2. BO結束定義: 任一條件不成立。
    3. PB定義: BO後，連續n根出現LL。


    """

    def __init__(self, params: ALBreakoutPullback50pctParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.bo_start_i = None  # 紀錄breakout開始的index
        self.bo_end_i = None  # 紀錄breakout結束的index
        self.pb_start_i = None  # 紀錄pullback開始的index
        self.pb_target_price = None  # pullback進場價格
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
            "strong_bar_series": ("body_strictly_increasing", n),
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

        ## 亞、歐、美盤前盤後不做
        time_hour = df.index[i].hour
        cond_time_filter = time_hour not in [8,9,13,14,19,20,21,22,23,0,1]

        if i < 4:
            return intents
        # i為進場K線，i-1為反向K線R1，i-2為突破的最後一根K線



        
        # 無倉，檢查進場
        else:
            # 沒有倉位時，檢查狀態
            ## 1. state = None, BOEND, PB都要檢查BO

            if self.state in (None, "BOEND", "PB", "SBOEND", "SPB"):
                if len(self.p.pullback_ratio) == 2:
                    long_pullback_ratio = self.p.pullback_ratio[0]
                    short_pullback_ratio = self.p.pullback_ratio[1]
                else:
                    long_pullback_ratio = self.p.pullback_ratio[0]
                    short_pullback_ratio = self.p.pullback_ratio[0]
            
                hh = ctx.indicators["hh"]
                # 檢查前面是否為多頭突破

                ## 連續n根同向K線以上(BO)
                cond1 = streak_curr >= self.p.break_out_series_n
                ## 突破前高(BO、MC)
                cond2 = high_series.iat[i] > hh.iat[i - 1]
                ## 收盤大於MA(BO、MC)
                cond3 = close_series.iat[i] > ctx.indicators["ma"].iat[i]
                ## 策略條件做多或雙向
                cond4 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)
                ## 實體越來越強(BO)
                cond5 = ctx.indicators["strong_bar_series"].iat[i]
            

                # 檢查前面是否為空頭突破
                ll = ctx.indicators["ll"]
                
                ## 連續n根同向K線以上(BO)
                cond1_short = streak_curr <= -self.p.break_out_series_n
                ## 突破前低(BO)
                cond2_short = low_series.iat[i] < ll.iat[i - 1]
                ## 收盤小於MA(BO)
                cond3_short = close_series.iat[i] < ctx.indicators["ma"].iat[i]
                ## 策略條件做空或雙向
                cond4_short = (self.p.allow_side is None) or (self.p.allow_side == Side.SHORT)
                ## 實體越來越強(BO)
                cond5_short = ctx.indicators["strong_bar_series"].iat[i]

                if (cond1 and cond2 and cond3 and cond4 and cond5):
                    # 記錄breakout開始index
                    self.bo_start_i = i - streak_curr + 1
                    self.state = "BO"
                    

                elif (cond1_short and cond2_short and cond3_short and cond4_short and cond5_short):
                    # 記錄breakout開始index
                    self.bo_start_i = i - abs(streak_curr) + 1
                    self.state = "SBO"


            # 狀態已經是BO或MC，檢查什麼時候結束BO
            if self.state in ["BO"]:
                if len(self.p.pullback_ratio) == 2:
                    long_pullback_ratio = self.p.pullback_ratio[0]
                    short_pullback_ratio = self.p.pullback_ratio[1]
                else:
                    long_pullback_ratio = self.p.pullback_ratio[0]
                    short_pullback_ratio = self.p.pullback_ratio[0]
            
                hh = ctx.indicators["hh"]
                # 檢查前面是否為多頭突破

                ## 連續n根同向K線以上(BO)
                cond1 = streak_curr >= self.p.break_out_series_n
                ## 突破前高(BO、MC)
                cond2 = high_series.iat[i] > hh.iat[i - 1]
                ## 收盤大於MA(BO、MC)
                cond3 = close_series.iat[i] > ctx.indicators["ma"].iat[i]
                ## 策略條件做多或雙向
                cond4 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)
                ## 實體越來越強(BO)
                cond5 = ctx.indicators["strong_bar_series"].iat[i]
            
                ## 任一條件不成立就結束BO
                if not all([cond1, cond2, cond3, cond4, cond5]):
                    ## 撿查BO是否合格
                    bo_start_price = open_series.iat[self.bo_start_i] if self.bo_start_i > 0 else float("nan")
                    bo_end_price = close_series.iat[i-1]
                    bo_range = abs(bo_end_price - bo_start_price)
                    atr = ctx.indicators["atr"].iat[i]
                    # 幅度需大於ATR倍數
                    range_cond1 = bo_range > atr * self.p.BO_n_times_atr
                    # 幅度需大於固定百分比
                    range_cond2 = bo_range > bo_start_price * self.p.min_sl_range_pct/100
                    if range_cond1 and range_cond2:
                        self.state = "BOEND"
                        self.bo_end_i = i - 1
                        
                    else:
                        self.state = None
                        self.bo_start_i = None
                        self.bo_end_i = None
                        self.pb_start_i = None
                        self.pb_target_price = None
                    
            elif self.state in ["SBO"]:
                # 檢查前面是否為空頭突破
                ll = ctx.indicators["ll"]
                
                ## 連續n根同向K線以上(BO)
                cond1_short = streak_curr <= -self.p.break_out_series_n
                ## 突破前低(BO)
                cond2_short = low_series.iat[i] < ll.iat[i - 1]
                ## 收盤小於MA(BO)
                cond3_short = close_series.iat[i] < ctx.indicators["ma"].iat[i]
                ## 策略條件做空或雙向
                cond4_short = (self.p.allow_side is None) or (self.p.allow_side == Side.SHORT)
                ## 實體越來越強(BO)
                cond5_short = ctx.indicators["strong_bar_series"].iat[i]
                ## 任一條件不成立就結束BO
                if not all([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short]):
                    ## 撿查BO是否合格，合格的話進入SBOEND狀態檢查是否有符合的PB
                    bo_start_price = open_series.iat[self.bo_start_i] if self.bo_start_i > 0 else float("nan")
                    bo_end_price = close_series.iat[i-1]
                    bo_range = abs(bo_end_price - bo_start_price)
                    atr = ctx.indicators["atr"].iat[i]
                    # 幅度需大於ATR倍數
                    range_cond1 = bo_range > atr * self.p.BO_n_times_atr
                    # 幅度需大於固定百分比
                    range_cond2 = bo_range > bo_start_price * self.p.min_sl_range_pct/100
                    if range_cond1 and range_cond2:
                        self.state = "SBOEND"
                        self.bo_end_i = i - 1
                        
                    else:
                        self.state = None
                        self.bo_start_i = None
                        self.bo_end_i = None
                        self.pb_start_i = None
                        self.pb_target_price = None

            elif self.state in ["BOEND", "SBOEND"]:
                ## BO結束後，檢查是否連續n根出現LL(多頭)或HH(空頭)，才代表合格PB
                ll_streak = ctx.indicators["ll_streak"]
                hh_streak = ctx.indicators["hh_streak"]
                # BO結束到PB期間太久則重新檢查BO
                if (i - self.bo_end_i) > 5:
                    self.state = None
                    self.bo_start_i = None
                    self.bo_end_i = None
                    self.pb_start_i = None
                    self.pb_target_price = None
                if self.state == "BOEND":
                    if ll_streak.iat[i] >= 1:
                        self.state = "PB"
                        self.pb_start_i = i
                elif self.state == "SBOEND":
                    if hh_streak.iat[i] >= 1:
                        self.state = "SPB"
                        self.pb_start_i = i
                

            
            # 狀態已經是PB，檢查進場點
            elif self.state == "PB":
                # 計算bo 高低點 pullback ratio 位置
                bo_high = float(high_series.iloc[self.bo_start_i:self.pb_start_i+1].max())
                bo_low  = float(low_series.iloc[self.bo_start_i:self.pb_start_i+1].min())
                pb_level = bo_high - (bo_high - bo_low) * long_pullback_ratio
                # 檢查有沒有碰到pb_level
                long_cond_entry = low_series.iat[i] <= pb_level <= high_series.iat[i]
                if long_cond_entry and cond_time_filter:
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
                            entry_price=entry_price,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            be_price=None,
                            priority=10,
                        )
                    )
                    self.state = None
            elif self.state == "SPB":
                # 計算bo 高低點 pullback ratio 位置
                bo_high = float(high_series.iloc[self.bo_start_i:self.pb_start_i+1].max())
                bo_low  = float(low_series.iloc[self.bo_start_i:self.pb_start_i+1].min())
                pb_level = bo_low + (bo_high - bo_low) * short_pullback_ratio
                # 檢查有沒有碰到pb_level
                short_cond_entry = low_series.iat[i] <= pb_level <= high_series.iat[i]
                if short_cond_entry and cond_time_filter:
                    # 進場價格設在pb_level sell limit
                    entry_price = pb_level
                    # 停損設在channel上方
                    sl_price = bo_high
                    sl_distance = sl_price - entry_price
                    # 停利設在停損距離的RR倍
                    tp_price = entry_price - sl_distance * self.p.rr

                    # 計算可用資金與下單數量
                    max_notional_lose = base_equity * (self.p.max_notional_pct / 100)
                    qty = max_notional_lose / (abs(entry_price - sl_price)) if abs(entry_price - sl_price) > 0 else 0.0
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.SHORT,
                            qty=max(self.p.min_qty, float(qty)),
                            entry_price=entry_price,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            be_price=None,
                            priority=10,
                        )
                    )
                    self.state = None


        debug_info = {
            "state": self.state,

        }
        return intents
